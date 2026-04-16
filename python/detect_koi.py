#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from pillow_heif import register_heif_opener
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
except ImportError as exc:  # pragma: no cover - setup path
    print(
        "Missing Python dependencies. Install them with `.venv/bin/python -m pip install -r python/requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
KOI_LABELS = ["koi carp", "koi fish", "ornamental carp"]
DETECTION_MODES = {"auto", "fast", "balanced", "robust"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect koi carp and export crops plus JSON.")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output-root", required=True, help="Root output directory")
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-base",
        help="Transformers zero-shot object detection model id",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Generate fish masks and transparent cutouts in addition to crops",
    )
    parser.add_argument(
        "--segment-export-background-crop",
        action="store_true",
        help="Export a padded segmented crop composited onto a sampled uniform background color",
    )
    parser.add_argument(
        "--background-crop-gradient",
        action="store_true",
        help="Use a filtered multi-point gradient instead of one flat background color",
    )
    parser.add_argument(
        "--background-crop-gradient-grid",
        type=int,
        default=4,
        help="Gradient anchor grid size for background crop generation",
    )
    parser.add_argument(
        "--background-crop-gradient-blur",
        type=float,
        default=48.0,
        help="Blur radius used to smooth the generated background crop gradient",
    )
    parser.add_argument(
        "--background-crop-edge-softness",
        type=float,
        default=None,
        help="Feather radius for compositing the segmented fish onto the background crop",
    )
    parser.add_argument(
        "--segmentation-model-id",
        default="facebook/sam-vit-base",
        help="Model id or local directory for segmentation",
    )
    parser.add_argument(
        "--segmentation-min-score",
        type=float,
        default=0.88,
        help="Minimum SAM mask score to keep",
    )
    parser.add_argument(
        "--cutout-feather-radius",
        type=float,
        default=0.0,
        help="Feather cutout alpha edges by this many pixels",
    )
    parser.add_argument(
        "--cutout-store-background-color",
        action="store_true",
        help="Sample and store a recommended background color around each mask",
    )
    parser.add_argument(
        "--cutout-background-ring",
        type=int,
        default=12,
        help="Ring width in pixels used for background color sampling",
    )
    parser.add_argument(
        "--cutout-decontaminate-edges",
        action="store_true",
        help="Remove background color spill from semi-transparent cutout edges",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=str(Path(".model-cache").resolve()),
        help="Local cache directory for model files",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require model files to be loaded from local cache only",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "fast", "balanced", "robust"],
        default="auto",
        help="Detection strategy preset",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Detection threshold")
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Text threshold when supported by the selected processor",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=0.35,
        help="Drop detections larger than this fraction of the image area",
    )
    parser.add_argument(
        "--crop-pad-x",
        type=float,
        default=0.0,
        help="Extra horizontal crop space per side as a fraction of box width",
    )
    parser.add_argument(
        "--crop-pad-y",
        type=float,
        default=0.0,
        help="Extra vertical crop space per side as a fraction of box height",
    )
    parser.add_argument(
        "--extend-crop-canvas",
        action="store_true",
        help="Extend crop beyond image edges using replicated edge pixels",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated file extensions to include, for example .heic,.heif",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=960,
        help="Tile size for multi-pass detection on larger images",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Fractional overlap between adjacent tiles",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Torch device override",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def normalize_extensions(raw_extensions: str | None) -> set[str] | None:
    if raw_extensions is None:
        return None
    extensions = {
        extension.strip().lower() if extension.strip().startswith(".") else f".{extension.strip().lower()}"
        for extension in raw_extensions.split(",")
        if extension.strip()
    }
    if not extensions:
        raise ValueError("At least one extension must be provided to --extensions.")
    unsupported = extensions - SUPPORTED_EXTENSIONS
    if unsupported:
        unsupported_list = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported extensions requested: {unsupported_list}")
    return extensions


def is_allowed_image(path: Path, allowed_extensions: set[str] | None) -> bool:
    if not is_supported_image(path):
        return False
    if allowed_extensions is None:
        return True
    return path.suffix.lower() in allowed_extensions


def collect_images(
    input_path: Path,
    limit: int | None = None,
    allowed_extensions: set[str] | None = None,
) -> list[Path]:
    if is_allowed_image(input_path, allowed_extensions):
        images = [input_path]
        return images[:limit] if limit is not None else images
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input must be a supported image file or directory: {input_path}")
    images = sorted(path for path in input_path.rglob("*") if is_allowed_image(path, allowed_extensions))
    return images[:limit] if limit is not None else images


def clamp_box(box: list[float], width: int, height: int) -> dict[str, int]:
    xmin = max(0, min(width - 1, math.floor(box[0])))
    ymin = max(0, min(height - 1, math.floor(box[1])))
    xmax = max(xmin + 1, min(width, math.ceil(box[2])))
    ymax = max(ymin + 1, min(height, math.ceil(box[3])))
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def box_area(box: dict[str, int]) -> int:
    return max(0, box["xmax"] - box["xmin"]) * max(0, box["ymax"] - box["ymin"])


def intersection_over_union(left: dict[str, int], right: dict[str, int]) -> float:
    xmin = max(left["xmin"], right["xmin"])
    ymin = max(left["ymin"], right["ymin"])
    xmax = min(left["xmax"], right["xmax"])
    ymax = min(left["ymax"], right["ymax"])

    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    if intersection == 0:
        return 0.0

    union = box_area(left) + box_area(right) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def apply_nms(detections: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if all(
            intersection_over_union(candidate["boundingBox"], existing["boundingBox"]) < iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def detection_signature(detection: dict[str, Any]) -> tuple[int, int, int, int]:
    box = detection["boundingBox"]
    return (box["xmin"], box["ymin"], box["xmax"], box["ymax"])


def dedupe_exact_boxes(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_box: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for detection in detections:
        key = detection_signature(detection)
        current = best_by_box.get(key)
        if current is None or detection["confidence"] > current["confidence"]:
            best_by_box[key] = detection
    return list(best_by_box.values())


def relative_stem(base_path: Path, image_path: Path) -> Path:
    if base_path.is_file():
        return Path(image_path.stem)
    relative = image_path.relative_to(base_path)
    return relative.parent / relative.stem


def build_crop_filename(image_path: Path, box: dict[str, int]) -> str:
    suffix = image_path.suffix.lower() if image_path.suffix.lower() in SUPPORTED_EXTENSIONS else ".png"
    return (
        f"{image_path.stem}__"
        f"x{box['xmin']}_y{box['ymin']}_x{box['xmax']}_y{box['ymax']}"
        f"{suffix}"
    )


def build_crop_box(
    bounding_box: dict[str, int],
    image_width: int,
    image_height: int,
    crop_pad_x: float,
    crop_pad_y: float,
) -> tuple[dict[str, int], dict[str, int]]:
    box_width = bounding_box["xmax"] - bounding_box["xmin"]
    box_height = bounding_box["ymax"] - bounding_box["ymin"]
    pad_x = math.ceil(box_width * crop_pad_x)
    pad_y = math.ceil(box_height * crop_pad_y)

    requested = {
        "xmin": bounding_box["xmin"] - pad_x,
        "ymin": bounding_box["ymin"] - pad_y,
        "xmax": bounding_box["xmax"] + pad_x,
        "ymax": bounding_box["ymax"] + pad_y,
    }
    clipped = {
        "xmin": max(0, requested["xmin"]),
        "ymin": max(0, requested["ymin"]),
        "xmax": min(image_width, requested["xmax"]),
        "ymax": min(image_height, requested["ymax"]),
    }
    return requested, clipped


def render_crop(
    image: Image.Image,
    requested_crop_box: dict[str, int],
    clipped_crop_box: dict[str, int],
    extend_crop_canvas: bool,
) -> Image.Image:
    if not extend_crop_canvas:
        return image.crop(
            (
                clipped_crop_box["xmin"],
                clipped_crop_box["ymin"],
                clipped_crop_box["xmax"],
                clipped_crop_box["ymax"],
            )
        )

    left_pad = max(0, -requested_crop_box["xmin"])
    top_pad = max(0, -requested_crop_box["ymin"])
    right_pad = max(0, requested_crop_box["xmax"] - image.width)
    bottom_pad = max(0, requested_crop_box["ymax"] - image.height)

    if left_pad == 0 and top_pad == 0 and right_pad == 0 and bottom_pad == 0:
        return image.crop(
            (
                requested_crop_box["xmin"],
                requested_crop_box["ymin"],
                requested_crop_box["xmax"],
                requested_crop_box["ymax"],
            )
        )

    array = np.asarray(image)
    padded = np.pad(
        array,
        ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
        mode="edge",
    )
    padded_image = Image.fromarray(padded, mode="RGB")
    return padded_image.crop(
        (
            requested_crop_box["xmin"] + left_pad,
            requested_crop_box["ymin"] + top_pad,
            requested_crop_box["xmax"] + left_pad,
            requested_crop_box["ymax"] + top_pad,
        )
    )


def prepare_inputs_for_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    prepared_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if getattr(value, "dtype", None) == torch.float64:
                value = value.to(dtype=torch.float32)
            prepared_inputs[key] = value.to(device)
        else:
            prepared_inputs[key] = value
    return prepared_inputs


def generate_tiles(image: Image.Image, tile_size: int, tile_overlap: float) -> list[tuple[str, Image.Image, int, int]]:
    if tile_size <= 0:
        raise ValueError("--tile-size must be a positive integer.")
    if not 0 <= tile_overlap < 1:
        raise ValueError("--tile-overlap must be between 0 and 1.")

    width, height = image.size
    if width <= tile_size and height <= tile_size:
        return [("full", image, 0, 0)]

    stride = max(1, int(tile_size * (1 - tile_overlap)))
    max_x = max(0, width - tile_size)
    max_y = max(0, height - tile_size)

    x_positions = list(range(0, max_x + 1, stride))
    y_positions = list(range(0, max_y + 1, stride))
    if x_positions[-1] != max_x:
        x_positions.append(max_x)
    if y_positions[-1] != max_y:
        y_positions.append(max_y)

    tiles: list[tuple[str, Image.Image, int, int]] = [("full", image, 0, 0)]
    for y in y_positions:
        for x in x_positions:
            if x == 0 and y == 0 and width <= tile_size and height <= tile_size:
                continue
            right = min(width, x + tile_size)
            lower = min(height, y + tile_size)
            tile_name = f"tile_x{x}_y{y}"
            tiles.append((tile_name, image.crop((x, y, right, lower)), x, y))
    return tiles


def make_highlight_suppressed_variant(image: Image.Image) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 255.0
    max_channel = array.max(axis=2)
    min_channel = array.min(axis=2)
    saturation = max_channel - min_channel
    luma = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]

    highlight_mask = (luma > 0.72) & (saturation < 0.28)
    compressed = array.copy()
    compressed[highlight_mask] *= 0.76

    low = np.percentile(compressed, 1.5, axis=(0, 1), keepdims=True)
    high = np.percentile(compressed, 98.5, axis=(0, 1), keepdims=True)
    normalized = np.clip((compressed - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
    variant = Image.fromarray((normalized * 255).astype(np.uint8), mode="RGB")
    variant = ImageEnhance.Contrast(variant).enhance(1.12)
    variant = ImageEnhance.Color(variant).enhance(1.08)
    return variant


def build_detection_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    autocontrast = ImageOps.autocontrast(image, cutoff=1)
    autocontrast = ImageEnhance.Contrast(autocontrast).enhance(1.18)
    autocontrast = ImageEnhance.Sharpness(autocontrast).enhance(1.1)

    median = image.filter(ImageFilter.MedianFilter(size=3))
    median = ImageOps.autocontrast(median, cutoff=1)
    median = ImageEnhance.Contrast(median).enhance(1.1)

    return [
        ("original", image),
        ("autocontrast", autocontrast),
        ("median_autocontrast", median),
        ("highlight_suppressed", make_highlight_suppressed_variant(image)),
    ]


def resolve_detection_mode(requested_mode: str, image_path: Path) -> str:
    if requested_mode != "auto":
        return requested_mode
    if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return "fast"
    return "balanced"


def select_detection_variants(image: Image.Image, detection_mode: str) -> list[tuple[str, Image.Image]]:
    if detection_mode == "fast":
        return [("original", image)]

    autocontrast = ImageOps.autocontrast(image, cutoff=1)
    autocontrast = ImageEnhance.Contrast(autocontrast).enhance(1.18)
    autocontrast = ImageEnhance.Sharpness(autocontrast).enhance(1.1)

    if detection_mode == "balanced":
        return [
            ("original", image),
            ("autocontrast", autocontrast),
        ]

    median = image.filter(ImageFilter.MedianFilter(size=3))
    median = ImageOps.autocontrast(median, cutoff=1)
    median = ImageEnhance.Contrast(median).enhance(1.1)

    return [
        ("original", image),
        ("autocontrast", autocontrast),
        ("median_autocontrast", median),
        ("highlight_suppressed", make_highlight_suppressed_variant(image)),
    ]


def select_tiles(
    image: Image.Image,
    tile_size: int,
    tile_overlap: float,
    detection_mode: str,
) -> list[tuple[str, Image.Image, int, int]]:
    if detection_mode == "fast":
        return [("full", image, 0, 0)]
    if detection_mode == "balanced" and max(image.width, image.height) <= tile_size * 1.5:
        return [("full", image, 0, 0)]
    return generate_tiles(image, tile_size, tile_overlap)


def run_single_pass(
    model: AutoModelForZeroShotObjectDetection,
    processor: AutoProcessor,
    image: Image.Image,
    labels: list[str],
    box_threshold: float,
    text_threshold: float,
    max_box_area_ratio: float,
    device: str,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    image_area = image.width * image.height

    for label in labels:
        inputs = processor(text=[label], images=image, return_tensors="pt")
        prepared_inputs = prepare_inputs_for_device(inputs, device)

        with torch.inference_mode():
            outputs = model(**prepared_inputs)

        post_process = processor.post_process_grounded_object_detection
        signature = inspect.signature(post_process)
        kwargs: dict[str, Any] = {
            "outputs": outputs,
            "threshold": box_threshold,
            "target_sizes": [(image.height, image.width)],
            "text_labels": [label],
        }
        if "text_threshold" in signature.parameters:
            kwargs["text_threshold"] = text_threshold

        results = post_process(**kwargs)[0]

        scores = results["scores"].tolist()
        boxes = results["boxes"].tolist()
        raw_text_labels = results.get("text_labels")

        if raw_text_labels is None:
            text_labels = [label] * len(boxes)
        elif isinstance(raw_text_labels, str):
            text_labels = [raw_text_labels] * len(boxes)
        else:
            text_labels = [str(item) for item in raw_text_labels]

        if len(text_labels) == 1 and len(boxes) > 1:
            text_labels = text_labels * len(boxes)
        elif len(text_labels) != len(boxes):
            text_labels = [label] * len(boxes)

        pair_count = min(len(scores), len(boxes))
        if pair_count == 0:
            continue

        if len(scores) != len(boxes):
            print(
                (
                    "Warning: detector output length mismatch for "
                    f"{label}: scores={len(scores)} boxes={len(boxes)} "
                    f"text_labels={len(text_labels)}. Truncating to {pair_count}."
                ),
                file=sys.stderr,
            )

        for index in range(pair_count):
            score = scores[index]
            text_label = text_labels[index] if index < len(text_labels) else label
            raw_box = boxes[index]
            bounding_box = clamp_box(raw_box, image.width, image.height)
            if box_area(bounding_box) / image_area > max_box_area_ratio:
                continue

            detections.append(
                {
                    "label": "koi carp",
                    "matchedPrompt": text_label,
                    "confidence": round(float(score), 6),
                    "boundingBox": bounding_box,
                    "shapeAssessment": {
                        "isStraight": None,
                        "status": "not_evaluated",
                    },
                }
            )

    return detections


def offset_detection(detection: dict[str, Any], x_offset: int, y_offset: int) -> dict[str, Any]:
    box = detection["boundingBox"]
    adjusted = dict(detection)
    adjusted["boundingBox"] = {
        "xmin": box["xmin"] + x_offset,
        "ymin": box["ymin"] + y_offset,
        "xmax": box["xmax"] + x_offset,
        "ymax": box["ymax"] + y_offset,
    }
    return adjusted


def run_inference(
    model: AutoModelForZeroShotObjectDetection,
    processor: AutoProcessor,
    image_path: Path,
    image: Image.Image,
    labels: list[str],
    mode: str,
    box_threshold: float,
    text_threshold: float,
    max_box_area_ratio: float,
    tile_size: int,
    tile_overlap: float,
    device: str,
    progress_label: str,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    detection_mode = resolve_detection_mode(mode, image_path)
    variants = select_detection_variants(image, detection_mode)
    total_variants = len(variants)

    print(
        f"{progress_label} mode: {detection_mode}",
        file=sys.stderr,
    )
    for variant_index, (variant_name, variant_image) in enumerate(variants, start=1):
        tiles = select_tiles(variant_image, tile_size, tile_overlap, detection_mode)
        print(
            (
                f"{progress_label} variant {variant_index}/{total_variants}: "
                f"{variant_name} with {len(tiles)} tile(s)"
            ),
            file=sys.stderr,
        )
        for tile_index, (tile_name, tile_image, x_offset, y_offset) in enumerate(tiles, start=1):
            if len(tiles) > 1:
                print(
                    (
                        f"{progress_label} tile {tile_index}/{len(tiles)} "
                        f"for {variant_name}: {tile_name}"
                    ),
                    file=sys.stderr,
                )
            pass_detections = run_single_pass(
                model=model,
                processor=processor,
                image=tile_image,
                labels=labels,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_box_area_ratio=max_box_area_ratio,
                device=device,
            )
            for detection in pass_detections:
                adjusted = offset_detection(detection, x_offset, y_offset)
                adjusted["sourcePass"] = {
                    "variant": variant_name,
                    "tile": tile_name,
                    "tileOrigin": {"x": x_offset, "y": y_offset},
                }
                detections.append(adjusted)

    return apply_nms(dedupe_exact_boxes(detections), iou_threshold=0.45)


def encode_mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def compute_mask_bbox(mask: np.ndarray) -> dict[str, int] | None:
    foreground = np.argwhere(mask)
    if foreground.size == 0:
        return None
    ymin, xmin = foreground.min(axis=0)
    ymax, xmax = foreground.max(axis=0)
    return {
        "xmin": int(xmin),
        "ymin": int(ymin),
        "xmax": int(xmax) + 1,
        "ymax": int(ymax) + 1,
    }


def apply_mask_cutout(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    rgba.putalpha(alpha)
    return rgba


def build_alpha_mask(mask: np.ndarray, feather_radius: float) -> Image.Image:
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return alpha


def extract_background_samples(
    image: Image.Image,
    mask: np.ndarray,
    ring_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if ring_width <= 0:
        raise ValueError("ring_width must be positive")

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    expanded = mask_image.filter(ImageFilter.MaxFilter(size=(ring_width * 2) + 1))
    expanded_mask = np.asarray(expanded) > 0
    ring_mask = expanded_mask & ~mask

    sampled = np.asarray(image)[ring_mask]
    positions = np.argwhere(ring_mask)
    return sampled, positions


def rgb_to_hsv_vectors(sampled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = sampled.astype(np.float32) / 255.0
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    saturation = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)
    hue = np.zeros_like(maxc)

    mask = delta > 1e-6
    red_is_max = mask & (maxc == r)
    green_is_max = mask & (maxc == g)
    blue_is_max = mask & (maxc == b)

    hue[red_is_max] = ((g[red_is_max] - b[red_is_max]) / delta[red_is_max]) % 6.0
    hue[green_is_max] = ((b[green_is_max] - r[green_is_max]) / delta[green_is_max]) + 2.0
    hue[blue_is_max] = ((r[blue_is_max] - g[blue_is_max]) / delta[blue_is_max]) + 4.0
    hue = (hue / 6.0) % 1.0

    return hue, saturation, maxc


def filter_background_samples(
    sampled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if sampled.size == 0:
        empty_mask = np.zeros(0, dtype=bool)
        return sampled, empty_mask, {
            "status": "empty_ring",
            "rawSampledPixelCount": 0,
            "filteredPixelCount": 0,
        }

    hue, saturation, value = rgb_to_hsv_vectors(sampled)

    bright_limit = float(np.percentile(value, 88))
    hard_bright_limit = float(np.percentile(value, 97))
    highlight_mask = (value >= hard_bright_limit) | ((value >= bright_limit) & (saturation <= 0.18))

    keep_mask = ~highlight_mask
    minimum_keep = max(32, sampled.shape[0] // 20)
    if int(keep_mask.sum()) < minimum_keep:
        keep_mask = np.ones(sampled.shape[0], dtype=bool)

    dominant_hue_applied = False
    colorful_mask = keep_mask & (saturation >= 0.12) & (value >= 0.08)
    if int(colorful_mask.sum()) >= 64:
        colorful_hue = hue[colorful_mask]
        hist, _ = np.histogram(colorful_hue, bins=12, range=(0.0, 1.0))
        dominant_bin = int(np.argmax(hist))
        bin_width = 1.0 / 12.0
        dominant_center = (dominant_bin + 0.5) * bin_width
        hue_distance = np.abs(((hue - dominant_center + 0.5) % 1.0) - 0.5)
        hue_keep = hue_distance <= (bin_width * 1.5)
        shadow_limit = float(np.percentile(value[keep_mask], 35))
        shadow_keep = keep_mask & ((value <= shadow_limit) | ((saturation < 0.12) & (value < 0.30)))
        refined_mask = keep_mask & (hue_keep | shadow_keep)
        if int(refined_mask.sum()) >= max(24, int(keep_mask.sum()) // 5):
            keep_mask = refined_mask
            dominant_hue_applied = True

    filtered = sampled[keep_mask]
    return filtered, keep_mask, {
        "status": "ok",
        "rawSampledPixelCount": int(sampled.shape[0]),
        "filteredPixelCount": int(filtered.shape[0]),
        "highlightExcludedPixelCount": int(highlight_mask.sum()),
        "dominantHueFilterApplied": dominant_hue_applied,
        "filterProfile": "water_gradient_v1",
    }


def compute_background_color(
    image: Image.Image,
    mask: np.ndarray,
    ring_width: int,
) -> dict[str, Any]:
    sampled, _positions = extract_background_samples(image, mask, ring_width)
    if sampled.size == 0:
        return {
            "status": "empty_ring",
            "ringWidth": ring_width,
            "sampledPixelCount": 0,
        }

    filtered, _keep_mask, filter_meta = filter_background_samples(sampled)
    if filtered.size == 0:
        filtered = sampled

    mean_rgb = filtered.mean(axis=0)
    median_rgb = np.median(filtered, axis=0)
    return {
        "status": "ok",
        "ringWidth": ring_width,
        "sampledPixelCount": int(sampled.shape[0]),
        "filteredPixelCount": int(filtered.shape[0]),
        "meanRgb": {
            "r": round(float(mean_rgb[0]), 2),
            "g": round(float(mean_rgb[1]), 2),
            "b": round(float(mean_rgb[2]), 2),
        },
        "medianRgb": {
            "r": round(float(median_rgb[0]), 2),
            "g": round(float(median_rgb[1]), 2),
            "b": round(float(median_rgb[2]), 2),
        },
        "recommendedCanvasColor": {
            "r": round(float(median_rgb[0]), 2),
            "g": round(float(median_rgb[1]), 2),
            "b": round(float(median_rgb[2]), 2),
        },
        "samplingFilter": filter_meta,
    }


def get_recommended_background_rgb(background_color: dict[str, Any] | None) -> np.ndarray | None:
    if not background_color or background_color.get("status") != "ok":
        return None
    recommended = background_color.get("recommendedCanvasColor")
    if not isinstance(recommended, dict):
        return None
    return np.array(
        [
            float(recommended["r"]),
            float(recommended["g"]),
            float(recommended["b"]),
        ],
        dtype=np.float32,
    )


def quantize_rgb(rgb: np.ndarray) -> tuple[int, int, int]:
    return (
        int(round(float(rgb[0]))),
        int(round(float(rgb[1]))),
        int(round(float(rgb[2]))),
    )


def build_rgb_payload(rgb: tuple[int, int, int]) -> dict[str, int]:
    return {"r": rgb[0], "g": rgb[1], "b": rgb[2]}


def compute_region_median_rgb(
    image: Image.Image,
    region_box: dict[str, int],
) -> tuple[int, int, int]:
    region = image.crop(
        (
            region_box["xmin"],
            region_box["ymin"],
            region_box["xmax"],
            region_box["ymax"],
        )
    )
    pixels = np.asarray(region)
    if pixels.size == 0:
        pixels = np.asarray(image)
    median_rgb = np.median(pixels.reshape(-1, 3), axis=0)
    return quantize_rgb(median_rgb)


def resolve_background_fill(
    image: Image.Image,
    background_color: dict[str, Any] | None,
    fallback_region_box: dict[str, int],
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    recommended_rgb = get_recommended_background_rgb(background_color)
    if recommended_rgb is not None:
        fill_rgb = quantize_rgb(recommended_rgb)
        return fill_rgb, {
            "status": "ok",
            "source": "recommendedCanvasColor",
            "fillColor": build_rgb_payload(fill_rgb),
        }

    fill_rgb = compute_region_median_rgb(image, fallback_region_box)
    return fill_rgb, {
        "status": "fallback",
        "source": "cropMedianFallback",
        "fillColor": build_rgb_payload(fill_rgb),
    }


def downsample_samples(
    positions: np.ndarray,
    colors: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if colors.shape[0] <= max_samples:
        return positions, colors

    step = max(1, int(math.ceil(colors.shape[0] / max_samples)))
    return positions[::step], colors[::step]


def create_gradient_background_layer(
    output_width: int,
    output_height: int,
    local_positions: np.ndarray,
    sampled_colors: np.ndarray,
    fallback_rgb: tuple[int, int, int],
    grid_size: int,
    blur_radius: float,
) -> Image.Image:
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

    if sampled_colors.size == 0:
        grid[:, :] = np.array(fallback_rgb, dtype=np.float32)
    else:
        positions, colors = downsample_samples(local_positions.astype(np.float32), sampled_colors.astype(np.float32), 2048)
        for row in range(grid_size):
            center_y = ((row + 0.5) / grid_size) * output_height
            for column in range(grid_size):
                center_x = ((column + 0.5) / grid_size) * output_width
                deltas = positions - np.array([center_y, center_x], dtype=np.float32)
                distances = np.sum(deltas * deltas, axis=1)
                neighbor_count = min(128, colors.shape[0])
                nearest_indices = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
                nearest_distances = distances[nearest_indices]
                weights = 1.0 / np.maximum(nearest_distances, 1.0)
                grid[row, column] = np.average(colors[nearest_indices], axis=0, weights=weights)

    gradient = Image.fromarray(np.clip(grid, 0, 255).astype(np.uint8), mode="RGB").resize(
        (output_width, output_height),
        Image.BICUBIC,
    )
    if blur_radius > 0:
        gradient = gradient.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return gradient


def decontaminate_cutout_edges(
    rgba_image: Image.Image,
    background_rgb: np.ndarray,
) -> Image.Image:
    rgba = np.asarray(rgba_image).astype(np.float32)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4] / 255.0

    edge_mask = (alpha > 0) & (alpha < 0.999)
    if not np.any(edge_mask):
        return rgba_image

    safe_alpha = np.maximum(alpha, 1e-3)
    decontaminated = (rgb - background_rgb.reshape(1, 1, 3) * (1.0 - alpha)) / safe_alpha
    decontaminated = np.clip(decontaminated, 0, 255)
    rgb = np.where(edge_mask, decontaminated, rgb)

    output = np.concatenate([rgb, rgba[:, :, 3:4]], axis=2).astype(np.uint8)
    return Image.fromarray(output, mode="RGBA")


def apply_mask_cutout_with_options(
    image: Image.Image,
    mask: np.ndarray,
    feather_radius: float,
    background_color: dict[str, Any] | None,
    decontaminate_edges: bool,
) -> Image.Image:
    rgba = image.convert("RGBA")
    rgba.putalpha(build_alpha_mask(mask, feather_radius))
    if decontaminate_edges:
        background_rgb = get_recommended_background_rgb(background_color)
        if background_rgb is not None:
            rgba = decontaminate_cutout_edges(rgba, background_rgb)
    return rgba


def create_background_crop(
    image: Image.Image,
    mask: np.ndarray,
    cutout_rgba: Image.Image,
    requested_crop_box: dict[str, int],
    clipped_crop_box: dict[str, int],
    background_color: dict[str, Any] | None,
    ring_width: int,
    use_gradient: bool,
    gradient_grid: int,
    gradient_blur: float,
) -> tuple[Image.Image, dict[str, Any]]:
    output_width = requested_crop_box["xmax"] - requested_crop_box["xmin"]
    output_height = requested_crop_box["ymax"] - requested_crop_box["ymin"]
    fill_rgb, fill_meta = resolve_background_fill(
        image=image,
        background_color=background_color,
        fallback_region_box=clipped_crop_box,
    )

    if use_gradient:
        sampled, positions = extract_background_samples(image, mask, ring_width)
        filtered, keep_mask, filter_meta = filter_background_samples(sampled)
        filtered_positions = positions[keep_mask] if positions.size > 0 else positions
        if filtered.size == 0:
            filtered = sampled
            filtered_positions = positions

        local_positions = filtered_positions.astype(np.float32)
        if local_positions.size > 0:
            local_positions[:, 0] = np.clip(local_positions[:, 0] - requested_crop_box["ymin"], 0, output_height - 1)
            local_positions[:, 1] = np.clip(local_positions[:, 1] - requested_crop_box["xmin"], 0, output_width - 1)
        background_layer = create_gradient_background_layer(
            output_width=output_width,
            output_height=output_height,
            local_positions=local_positions,
            sampled_colors=filtered,
            fallback_rgb=fill_rgb,
            grid_size=gradient_grid,
            blur_radius=gradient_blur,
        )
        background_meta: dict[str, Any] = {
            "status": "ok",
            "style": "gradient",
            "fillColorSource": fill_meta["source"],
            "fallbackColor": fill_meta["fillColor"],
            "gradientGrid": gradient_grid,
            "gradientBlurRadius": round(float(gradient_blur), 2),
            "gradientSampleCount": int(filtered.shape[0]),
            "gradientSamplingFilter": filter_meta,
        }
    else:
        background_layer = Image.new("RGB", (output_width, output_height), fill_rgb)
        background_meta = {
            "status": "ok",
            "style": "flat",
            "fillColorSource": fill_meta["source"],
            "fillColor": fill_meta["fillColor"],
        }

    canvas = background_layer.convert("RGBA")
    patch = cutout_rgba.crop(
        (
            clipped_crop_box["xmin"],
            clipped_crop_box["ymin"],
            clipped_crop_box["xmax"],
            clipped_crop_box["ymax"],
        )
    )
    offset_x = clipped_crop_box["xmin"] - requested_crop_box["xmin"]
    offset_y = clipped_crop_box["ymin"] - requested_crop_box["ymin"]
    canvas.alpha_composite(patch, dest=(offset_x, offset_y))

    return canvas.convert("RGB"), {
        **background_meta,
        "requestedPaddingPreserved": True,
        "outputSize": {"width": output_width, "height": output_height},
    }


def segment_detection(
    image: Image.Image,
    bounding_box: dict[str, int],
    processor: SamProcessor,
    model: SamModel,
    device: str,
    min_score: float,
) -> dict[str, Any]:
    inputs = processor(
        images=image,
        input_boxes=[[[bounding_box["xmin"], bounding_box["ymin"], bounding_box["xmax"], bounding_box["ymax"]]]],
        return_tensors="pt",
    )
    original_sizes = inputs["original_sizes"].cpu()
    reshaped_input_sizes = inputs["reshaped_input_sizes"].cpu()
    prepared_inputs = prepare_inputs_for_device(inputs, device)

    with torch.inference_mode():
        outputs = model(**prepared_inputs)

    mask_batches = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes,
        reshaped_input_sizes,
    )
    iou_scores = outputs.iou_scores.detach().cpu().numpy()

    candidate_masks = mask_batches[0][0]
    candidate_scores = iou_scores[0][0]
    best_index = int(np.argmax(candidate_scores))
    best_score = float(candidate_scores[best_index])
    best_mask = candidate_masks[best_index].numpy().astype(bool)

    if best_score < min_score:
        return {
            "status": "rejected_low_score",
            "score": round(best_score, 6),
        }

    mask_bbox = compute_mask_bbox(best_mask)
    if mask_bbox is None:
        return {
            "status": "empty_mask",
            "score": round(best_score, 6),
        }

    return {
        "status": "ok",
        "score": round(best_score, 6),
        "mask": best_mask,
        "maskBoundingBox": mask_bbox,
        "foregroundAreaPixels": int(best_mask.sum()),
    }


def save_json(target_path: Path, payload: dict[str, Any]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_cache_dir = Path(args.model_cache_dir).expanduser().resolve()
    json_root = output_root / "json"
    crops_root = output_root / "crops"
    masks_root = output_root / "masks"
    cutouts_root = output_root / "cutouts"
    background_crops_root = output_root / "background-crops"
    manifest_path = output_root / "index.json"

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer.")
    if args.tile_size <= 0:
        raise SystemExit("--tile-size must be a positive integer.")
    if not 0 <= args.tile_overlap < 1:
        raise SystemExit("--tile-overlap must be between 0 and 1.")
    if args.crop_pad_x < 0 or args.crop_pad_y < 0:
        raise SystemExit("--crop-pad-x and --crop-pad-y must be non-negative.")
    if not 0 <= args.segmentation_min_score <= 1:
        raise SystemExit("--segmentation-min-score must be between 0 and 1.")
    if args.segment_export_background_crop and not args.segment:
        raise SystemExit("--segment-export-background-crop requires --segment.")
    if args.background_crop_gradient and not args.segment_export_background_crop:
        raise SystemExit("--background-crop-gradient requires --segment-export-background-crop.")
    if args.cutout_feather_radius < 0:
        raise SystemExit("--cutout-feather-radius must be non-negative.")
    if args.background_crop_edge_softness is not None and args.background_crop_edge_softness < 0:
        raise SystemExit("--background-crop-edge-softness must be non-negative.")
    if args.background_crop_gradient_grid < 2:
        raise SystemExit("--background-crop-gradient-grid must be at least 2.")
    if args.background_crop_gradient_blur < 0:
        raise SystemExit("--background-crop-gradient-blur must be non-negative.")
    if args.cutout_background_ring <= 0:
        raise SystemExit("--cutout-background-ring must be a positive integer.")
    allowed_extensions = normalize_extensions(args.extensions)
    background_crop_edge_softness = (
        args.background_crop_edge_softness
        if args.background_crop_edge_softness is not None
        else args.cutout_feather_radius
    )

    image_paths = collect_images(input_path, args.limit, allowed_extensions)
    if not image_paths:
        raise SystemExit("No supported images were found.")

    output_root.mkdir(parents=True, exist_ok=True)
    json_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    cutouts_root.mkdir(parents=True, exist_ok=True)
    background_crops_root.mkdir(parents=True, exist_ok=True)
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(
        (
            f"Loading model {args.model_id} on {device} "
            f"(cache: {model_cache_dir}, offline: {args.offline})..."
        ),
        file=sys.stderr,
    )
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            cache_dir=str(model_cache_dir),
            local_files_only=args.offline,
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.model_id,
            cache_dir=str(model_cache_dir),
            local_files_only=args.offline,
        ).to(device)
    except OSError as exc:
        if args.offline:
            raise SystemExit(
                (
                    "Offline mode could not find the model in the local cache. "
                    f"Run once with internet access first, or point --model to a local directory. "
                    f"Cache dir: {model_cache_dir}"
                )
            ) from exc
        raise
    model.eval()

    segmentation_processor: SamProcessor | None = None
    segmentation_model: SamModel | None = None
    if args.segment:
        print(
            (
                f"Loading segmentation model {args.segmentation_model_id} on {device} "
                f"(cache: {model_cache_dir}, offline: {args.offline})..."
            ),
            file=sys.stderr,
        )
        try:
            segmentation_processor = SamProcessor.from_pretrained(
                args.segmentation_model_id,
                cache_dir=str(model_cache_dir),
                local_files_only=args.offline,
            )
            segmentation_model = SamModel.from_pretrained(
                args.segmentation_model_id,
                cache_dir=str(model_cache_dir),
                local_files_only=args.offline,
            ).to(device)
        except OSError as exc:
            if args.offline:
                raise SystemExit(
                    (
                        "Offline mode could not find the segmentation model in the local cache. "
                        f"Run once with internet access first, or point --segmentation-model-id to a local directory. "
                        f"Cache dir: {model_cache_dir}"
                    )
                ) from exc
            raise
        segmentation_model.eval()

    images_payload: list[dict[str, Any]] = []
    detections_written = 0
    crops_written = 0
    background_crops_written = 0

    for image_path in image_paths:
        image_index = len(images_payload) + 1
        progress_label = f"[{image_index}/{len(image_paths)}] {image_path.name}"
        print(f"{progress_label} processing {image_path}...", file=sys.stderr)
        relative_json_stem = relative_stem(input_path, image_path)
        relative_parent = relative_json_stem.parent
        image_json_path = json_root / relative_json_stem.with_suffix(".json")
        image_crop_dir = crops_root / relative_parent

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            detections = run_inference(
                model=model,
                processor=processor,
                image_path=image_path,
                image=image,
                labels=KOI_LABELS,
                mode=args.mode,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                max_box_area_ratio=args.max_box_area_ratio,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                device=device,
                progress_label=progress_label,
            )

            image_payload = {
                "sourceImage": str(image_path),
                "relativeSourcePath": (
                    str(image_path.relative_to(input_path)) if input_path.is_dir() else image_path.name
                ),
                "imageSize": {"width": image.width, "height": image.height},
                "detector": {
                    "modelId": args.model_id,
                    "segmentationEnabled": args.segment,
                    "segmentExportBackgroundCrop": args.segment_export_background_crop if args.segment else None,
                    "backgroundCropGradient": args.background_crop_gradient if args.segment else None,
                    "backgroundCropGradientGrid": args.background_crop_gradient_grid if args.segment else None,
                    "backgroundCropGradientBlur": args.background_crop_gradient_blur if args.segment else None,
                    "backgroundCropEdgeSoftness": background_crop_edge_softness if args.segment else None,
                    "segmentationModelId": args.segmentation_model_id if args.segment else None,
                    "segmentationMinScore": args.segmentation_min_score if args.segment else None,
                    "cutoutFeatherRadius": args.cutout_feather_radius if args.segment else None,
                    "cutoutStoreBackgroundColor": args.cutout_store_background_color if args.segment else None,
                    "cutoutBackgroundRing": args.cutout_background_ring if args.segment else None,
                    "cutoutDecontaminateEdges": args.cutout_decontaminate_edges if args.segment else None,
                    "modelCacheDir": str(model_cache_dir),
                    "offline": args.offline,
                    "device": device,
                    "requestedMode": args.mode,
                    "candidateLabels": KOI_LABELS,
                    "cropPadX": args.crop_pad_x,
                    "cropPadY": args.crop_pad_y,
                    "extendCropCanvas": args.extend_crop_canvas,
                    "boxThreshold": args.box_threshold,
                    "textThreshold": args.text_threshold,
                    "maxBoxAreaRatio": args.max_box_area_ratio,
                    "tileSize": args.tile_size,
                    "tileOverlap": args.tile_overlap,
                },
                "detections": [],
            }

            image_crop_dir.mkdir(parents=True, exist_ok=True)
            image_mask_dir = masks_root / relative_parent
            image_cutout_dir = cutouts_root / relative_parent
            image_background_crop_dir = background_crops_root / relative_parent
            image_mask_dir.mkdir(parents=True, exist_ok=True)
            image_cutout_dir.mkdir(parents=True, exist_ok=True)
            image_background_crop_dir.mkdir(parents=True, exist_ok=True)
            for detection in detections:
                tight_box = detection["boundingBox"]
                requested_crop_box, clipped_crop_box = build_crop_box(
                    tight_box,
                    image.width,
                    image.height,
                    args.crop_pad_x,
                    args.crop_pad_y,
                )
                crop_filename = build_crop_filename(image_path, tight_box)
                crop_path = image_crop_dir / crop_filename
                crop = render_crop(
                    image=image,
                    requested_crop_box=requested_crop_box,
                    clipped_crop_box=clipped_crop_box,
                    extend_crop_canvas=args.extend_crop_canvas,
                )
                crop.save(crop_path)
                detection["cropRegion"] = {
                    "requestedBox": requested_crop_box,
                    "sourceBox": clipped_crop_box,
                    "padding": {
                        "horizontalRatioPerSide": args.crop_pad_x,
                        "verticalRatioPerSide": args.crop_pad_y,
                    },
                    "extendCropCanvas": args.extend_crop_canvas,
                    "outputSize": {"width": crop.width, "height": crop.height},
                }
                detection["cropImage"] = str(crop_path)

                if args.segment and segmentation_processor is not None and segmentation_model is not None:
                    segmentation = segment_detection(
                        image=image,
                        bounding_box=tight_box,
                        processor=segmentation_processor,
                        model=segmentation_model,
                        device=device,
                        min_score=args.segmentation_min_score,
                    )
                    if segmentation["status"] == "ok":
                        mask = segmentation.pop("mask")
                        mask_filename = f"{image_path.stem}__x{tight_box['xmin']}_y{tight_box['ymin']}_x{tight_box['xmax']}_y{tight_box['ymax']}__mask.png"
                        cutout_filename = f"{image_path.stem}__x{tight_box['xmin']}_y{tight_box['ymin']}_x{tight_box['xmax']}_y{tight_box['ymax']}__cutout.png"
                        background_crop_filename = f"{image_path.stem}__x{tight_box['xmin']}_y{tight_box['ymin']}_x{tight_box['xmax']}_y{tight_box['ymax']}__background-crop.jpg"
                        mask_path = image_mask_dir / mask_filename
                        cutout_path = image_cutout_dir / cutout_filename
                        background_crop_path = image_background_crop_dir / background_crop_filename
                        encode_mask_to_image(mask).save(mask_path)
                        background_color: dict[str, Any] | None = None
                        if (
                            args.cutout_store_background_color
                            or args.cutout_decontaminate_edges
                            or args.segment_export_background_crop
                        ):
                            background_color = compute_background_color(
                                image=image,
                                mask=mask,
                                ring_width=args.cutout_background_ring,
                            )
                        cutout_rgba = apply_mask_cutout_with_options(
                            image,
                            mask,
                            args.cutout_feather_radius,
                            background_color,
                            args.cutout_decontaminate_edges,
                        )
                        cutout_rgba.save(cutout_path)
                        segmentation["maskImage"] = str(mask_path)
                        segmentation["cutoutImage"] = str(cutout_path)
                        segmentation["cutoutFeatherRadius"] = args.cutout_feather_radius
                        segmentation["cutoutEdgeDecontaminated"] = args.cutout_decontaminate_edges
                        if args.segment_export_background_crop:
                            background_crop_cutout = cutout_rgba
                            if background_crop_edge_softness != args.cutout_feather_radius:
                                background_crop_cutout = apply_mask_cutout_with_options(
                                    image,
                                    mask,
                                    background_crop_edge_softness,
                                    background_color,
                                    args.cutout_decontaminate_edges,
                                )
                            background_crop, background_crop_meta = create_background_crop(
                                image=image,
                                mask=mask,
                                cutout_rgba=background_crop_cutout,
                                requested_crop_box=requested_crop_box,
                                clipped_crop_box=clipped_crop_box,
                                background_color=background_color,
                                ring_width=args.cutout_background_ring,
                                use_gradient=args.background_crop_gradient,
                                gradient_grid=args.background_crop_gradient_grid,
                                gradient_blur=args.background_crop_gradient_blur,
                            )
                            background_crop.save(background_crop_path, quality=95)
                            segmentation["backgroundCropImage"] = str(background_crop_path)
                            segmentation["backgroundCrop"] = background_crop_meta
                            segmentation["backgroundCropEdgeSoftness"] = background_crop_edge_softness
                            background_crops_written += 1
                        if (args.cutout_store_background_color or args.segment_export_background_crop) and background_color is not None:
                            segmentation["backgroundColor"] = background_color
                    detection["segmentation"] = segmentation
                else:
                    detection["segmentation"] = {"status": "not_requested"}
                image_payload["detections"].append(detection)

            detections_written += len(detections)
            crops_written += len(detections)
            images_payload.append(image_payload)
            save_json(image_json_path, image_payload)
            print(
                f"{progress_label} complete: {len(detections)} detection(s)",
                file=sys.stderr,
            )

    manifest_payload = {
        "sourceInput": str(input_path),
        "outputRoot": str(output_root),
        "extensions": sorted(allowed_extensions) if allowed_extensions is not None else None,
        "limit": args.limit,
        "imagesProcessed": len(image_paths),
        "detectionsWritten": detections_written,
        "cropsWritten": crops_written,
        "backgroundCropsWritten": background_crops_written,
        "images": images_payload,
    }
    save_json(manifest_path, manifest_payload)

    print(
        json.dumps(
            {
                "manifestPath": str(manifest_path),
                "imagesProcessed": len(image_paths),
                "detectionsWritten": detections_written,
                "cropsWritten": crops_written,
                "backgroundCropsWritten": background_crops_written,
                "modelId": args.model_id,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
