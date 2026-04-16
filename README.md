# Koi Detector

Detect koi carp in photos, write bounding boxes to JSON, and export one crop per detected fish.

Supported inputs:

- `jpg`
- `jpeg`
- `png`
- `webp`
- `bmp`
- `tif`
- `tiff`
- `heic`
- `heif`

## What it does

- Detects koi only
- Stores the tight fish bounding box as JSON
- Exports one crop image per detection
- Can optionally export one mask PNG and one transparent cutout PNG per detection
- Can optionally export a publication-style segmented crop on a uniform sampled background color
- Can optionally replace that flat background with a filtered water-tone gradient
- Can add extra crop space around the fish
- Can extend the crop beyond the original photo edge when needed
- Can filter by file extension inside mixed folders
- Can run from a local model cache without internet after the model is cached

The JSON format is also structured so a later step can add extra metadata such as `isStraight` for bent-vs-straight filtering.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r python/requirements.txt
pnpm install
```

## Repo Hygiene

This repo now avoids global `*.jpg` / `*.png` / `*.mp4` ignore rules so approved example media can be committed intentionally.

Current policy:

- generated detector outputs stay ignored under `output/` and `tmp/`
- raw or private source media should live in ignored folders such as `private/`, `private-media/`, or `photos-private/`
- local docs experiments should go in ignored folders such as `docs/private-assets/` or `docs/generated/`
- only reviewed, intentionally public assets should be copied into `docs/public-assets/`

That gives you a path to publish a few example images later without making the repo silently accept all local photo/video files.

## Model Cache And Offline Use

The detector stores model files in `./.model-cache` by default.

That means:

1. The first online run downloads the model into `./.model-cache`
2. Later runs can use `--offline`
3. You can also point `--model` at a local model directory instead of a Hugging Face model id

Example:

```bash
pnpm detect --input ./photos --output ./output
pnpm detect --input ./photos --output ./output --offline
```

If offline mode is enabled and the model is not present in the local cache, the script exits with a clear error.

## Basic Usage

```bash
pnpm detect --input ./photos --output ./output
```

JPG folder, fast path:

```bash
pnpm detect \
  --input ~/code/showPairsOrig/jpg \
  --output ./output \
  --extensions .jpg,.jpeg \
  --mode fast \
  --limit 25 \
  --device mps
```

JPG folder with segmentation enabled:

```bash
pnpm detect \
  --input ~/code/showPairsOrig/jpg \
  --output ./output \
  --extensions .jpg,.jpeg \
  --mode fast \
  --limit 25 \
  --device mps \
  --segment
```

JPG folder with softer cutout edges and edge spill cleanup:

```bash
pnpm detect \
  --input ~/code/showPairsOrig/jpg \
  --output ./output \
  --extensions .jpg,.jpeg \
  --mode fast \
  --limit 25 \
  --device mps \
  --segment \
  --cutout-feather-radius 2.5 \
  --cutout-store-background-color \
  --cutout-background-ring 16 \
  --cutout-decontaminate-edges
```

JPG folder with a flattened segmented crop on a sampled uniform background:

```bash
pnpm detect \
  --input ~/code/showPairsOrig/jpg \
  --output ./output \
  --extensions .jpg,.jpeg \
  --mode fast \
  --limit 25 \
  --device mps \
  --segment \
  --crop-pad-x 0.20 \
  --crop-pad-y 0.10 \
  --segment-export-background-crop \
  --cutout-background-ring 16 \
  --cutout-decontaminate-edges
```

JPG folder with a softer pasted edge and filtered gradient background:

```bash
pnpm detect \
  --input ~/code/showPairsOrig/jpg \
  --output ./output \
  --extensions .jpg,.jpeg \
  --mode fast \
  --limit 25 \
  --device mps \
  --segment \
  --crop-pad-x 0.20 \
  --crop-pad-y 0.10 \
  --segment-export-background-crop \
  --background-crop-gradient \
  --background-crop-gradient-grid 5 \
  --background-crop-gradient-blur 56 \
  --background-crop-edge-softness 3.5 \
  --cutout-background-ring 18 \
  --cutout-decontaminate-edges
```

HEIC folder, balanced path:

```bash
pnpm detect \
  --input ~/code/showPairsOrig \
  --output ./output \
  --extensions .heic,.heif \
  --mode balanced \
  --limit 25 \
  --device mps
```

Looser crops with synthetic edge extension:

```bash
pnpm detect \
  --input ~/code/showPairsOrig/jpg \
  --output ./output \
  --extensions .jpg,.jpeg \
  --mode fast \
  --limit 6 \
  --device mps \
  --crop-pad-x 0.20 \
  --crop-pad-y 0.10 \
  --extend-crop-canvas
```

Use a custom model cache directory:

```bash
pnpm detect \
  --input ./photos \
  --output ./output \
  --model-cache-dir ./my-model-cache
```

Use a local model directory instead of downloading:

```bash
pnpm detect \
  --input ./photos \
  --output ./output \
  --model /absolute/path/to/local-model \
  --offline
```

## Docs And GitHub Pages

There is now a docs skeleton in [docs/index.md](/Users/christianstampf/code/ai/fishDetect/docs/index.md:1).
There is also a checked-in sanitized demo JSON at [docs/public-assets/demo-json/IMG_6008.demo.json](/Users/christianstampf/code/ai/fishDetect/docs/public-assets/demo-json/IMG_6008.demo.json:1).

## Example Output

These examples are derived from `IMG_6008.jpg`. The original source photo is private and is not committed.

Tight crop example:

![IMG_6008 tight crop](docs/public-assets/examples/img-6008-tight-crop.jpg)

Gradient background publishing crop:

![IMG_6008 gradient background crop](docs/public-assets/examples/img-6008-gradient-background-crop.jpg)

Recommended workflow:

1. Generate local example outputs from one representative private image such as `IMG_6008.jpg` into `docs/generated/`
2. Review the outputs manually
3. Copy only approved, non-sensitive assets into `docs/public-assets/`
4. If you later enable GitHub Pages, publish from the `docs/` folder

That keeps raw photos private while still allowing a few curated examples in project docs.

Generate the standard single-image docs example set with:

```bash
pnpm docs:example -- --input ~/code/showPairsOrig/jpg/IMG_6008.jpg --device mps
```

That command is intentionally single-file only. It does not scan a folder, so it cannot accidentally pull all of your local photos into the docs workflow.

Use local directories for both detection and segmentation models:

```bash
pnpm detect \
  --input ./photos \
  --output ./output \
  --model /absolute/path/to/local-detector \
  --segmentation-model /absolute/path/to/local-segmentation-model \
  --segment \
  --offline
```

## Parameters

### Required

- `--input`
  File or directory to process.

### Output And Runtime

- `--output`
  Output root directory. Default: `./output`

- `--python-bin`
  Python executable to use. Default: `./.venv/bin/python` if present, otherwise `python3.12`

- `--device`
  Torch device selection.
  Allowed values:
  `auto`, `cpu`, `cuda`, `mps`

### Model And Caching

- `--model`
  Hugging Face model id or local model directory.
  Default: `IDEA-Research/grounding-dino-base`

- `--model-cache-dir`
  Local cache directory for model files.
  Default: `./.model-cache`

- `--offline`
  Forces model loading from local files only. No network fetches are allowed.

- `--segment`
  Enables segmentation. When enabled, the tool also writes:
  - one mask PNG per accepted segmentation
  - one transparent cutout PNG per accepted segmentation
  - optionally one flattened background crop per accepted segmentation
  - segmentation metadata into JSON

- `--segment-export-background-crop`
  Exports a padded segmented crop composited onto one uniform color sampled from the ring around the fish.
  This export preserves the full requested padded crop size even when the fish is near the image edge.
  The sampled background color metadata is also stored in JSON for this export.

- `--background-crop-gradient`
  Replaces the flat background crop fill with a smooth gradient built from multiple filtered samples around the fish.
  Bright reflections and bubble highlights are down-weighted by the sampling filter.

- `--background-crop-gradient-grid`
  Controls how many gradient anchor cells are used before smoothing.
  Higher values can preserve more local variation.
  Default: `4`

- `--background-crop-gradient-blur`
  Blur radius used to smooth the generated gradient.
  Higher values make the background calmer and less blotchy.
  Default: `48`

- `--background-crop-edge-softness`
  Feather radius used when pasting the segmented fish into the background crop.
  This is separate from `--cutout-feather-radius`, which controls the transparent cutout export.
  Default:
  falls back to `--cutout-feather-radius`

- `--segmentation-model`
  Hugging Face model id or local model directory for segmentation.
  Default: `facebook/sam-vit-base`

- `--segmentation-min-score`
  Minimum SAM score required to keep a segmentation result.
  Default: `0.88`

- `--cutout-feather-radius`
  Softens the transparent cutout alpha edge by the given pixel radius.
  Default: `0`

- `--cutout-store-background-color`
  Samples the ring of pixels just outside the fish mask and stores recommended background color metadata in JSON.

- `--cutout-background-ring`
  Width in pixels of the outside ring used for background color sampling.
  Default: `12`

- `--cutout-decontaminate-edges`
  Uses the sampled background color to reduce color spill on semi-transparent cutout edges.

Cutout behavior:

- These options only apply when `--segment` is enabled
- The regular box crop output still stays available
- `--cutout-decontaminate-edges` works best together with `--cutout-store-background-color`
- `--segment-export-background-crop` also uses the sampled ring color, and falls back to the crop median if the ring has no usable pixels
- The flattened background crop uses that sampled color as its whole background so pasted edges blend more naturally than a transparent cutout on a mismatched canvas
- `--background-crop-gradient` uses multiple nearby filtered samples instead of one flat fill, which helps keep the background in the same water color family while suppressing glare
- `--background-crop-edge-softness` is the main control for making the pasted fish edge less hard in the flattened publishing crop

### Detection Strategy

- `--mode`
  Detection preset.
  Allowed values:
  `auto`, `fast`, `balanced`, `robust`

Behavior:

- `auto`
  Uses `fast` for `jpg`, `jpeg`, `png`, `webp`, `bmp`
  Uses `balanced` for `heic`, `heif`, `tif`, `tiff`

- `fast`
  One pass on the original image only. Best for JPG batches and faster runs.

- `balanced`
  Runs a small number of enhancement passes. Better recovery than `fast` without the full cost of `robust`.

- `robust`
  Uses more enhancement variants and tiling. Best when glare, reflections, or clutter reduce detection quality.

### Detection Thresholds

- `--box-threshold`
  Confidence threshold for detections.
  Default: `0.25`

- `--text-threshold`
  Text matching threshold when supported by the selected processor.
  Default: `0.20`

- `--max-box-area-ratio`
  Rejects boxes larger than this fraction of the whole image.
  Default: `0.35`

This is used to suppress oversized group or scene-level boxes.

### Crop Controls

- `--crop-pad-x`
  Extra horizontal crop space per side, as a fraction of the tight bounding box width.
  Example:
  `0.20` means 20% extra space on the left and 20% extra space on the right.
  Default: `0`

- `--crop-pad-y`
  Extra vertical crop space per side, as a fraction of the tight bounding box height.
  Example:
  `0.10` means 10% extra space above and 10% extra space below.
  Default: `0`

- `--extend-crop-canvas`
  If the padded crop extends beyond the source image edge, replicate edge pixels so the output crop keeps the requested size instead of clipping.

Important:

- The stored `boundingBox` remains the tight fish box
- The exported crop can be larger than the tight box
- The padded crop region is stored separately in JSON under `cropRegion`

### Input Filtering

- `--extensions`
  Comma-separated extension filter inside a mixed folder.
  Example:
  `.heic,.heif`

- `--limit`
  Maximum number of images to process.

### Advanced Options

- `--tile-size`
  Tile size used by multi-pass detection on larger images.
  Default: `960`

- `--tile-overlap`
  Tile overlap fraction.
  Default: `0.25`

These are mainly useful for `balanced` and `robust`.

## Output Structure

The command creates:

- `output/index.json`
  Global manifest for the whole run

- `output/json/...`
  One JSON file per source image

- `output/crops/...`
  One crop image per detected koi

- `output/masks/...`
  One binary mask PNG per segmented koi when `--segment` is enabled

- `output/cutouts/...`
  One transparent PNG cutout per segmented koi when `--segment` is enabled

- `output/background-crops/...`
  One flattened JPG per segmented koi when `--segment-export-background-crop` is enabled

Crop filenames look like:

```text
<original-file-name>__x<xmin>_y<ymin>_x<xmax>_y<ymax>.<ext>
```

The filename uses the tight fish bounding box coordinates, not the padded crop region.

## Example JSON

Example per-image JSON shape:

```json
{
  "sourceImage": "/abs/path/pond.jpg",
  "relativeSourcePath": "pond.jpg",
  "imageSize": {
    "width": 2400,
    "height": 1600
  },
  "detector": {
    "modelId": "IDEA-Research/grounding-dino-base",
    "segmentationEnabled": true,
    "segmentExportBackgroundCrop": true,
    "backgroundCropGradient": true,
    "backgroundCropGradientGrid": 5,
    "backgroundCropGradientBlur": 56,
    "backgroundCropEdgeSoftness": 3.5,
    "segmentationModelId": "facebook/sam-vit-base",
    "segmentationMinScore": 0.88,
    "cutoutFeatherRadius": 2.5,
    "cutoutStoreBackgroundColor": true,
    "cutoutBackgroundRing": 16,
    "cutoutDecontaminateEdges": true,
    "modelCacheDir": "/abs/path/.model-cache",
    "offline": false,
    "device": "mps",
    "requestedMode": "auto",
    "candidateLabels": ["koi carp", "koi fish", "ornamental carp"],
    "cropPadX": 0.2,
    "cropPadY": 0.1,
    "extendCropCanvas": true,
    "boxThreshold": 0.25,
    "textThreshold": 0.2,
    "maxBoxAreaRatio": 0.35,
    "tileSize": 960,
    "tileOverlap": 0.25
  },
  "detections": [
    {
      "label": "koi carp",
      "matchedPrompt": "koi fish",
      "confidence": 0.91,
      "boundingBox": {
        "xmin": 812,
        "ymin": 401,
        "xmax": 1288,
        "ymax": 723
      },
      "shapeAssessment": {
        "isStraight": null,
        "status": "not_evaluated"
      },
      "cropRegion": {
        "requestedBox": {
          "xmin": 717,
          "ymin": 369,
          "xmax": 1383,
          "ymax": 755
        },
        "sourceBox": {
          "xmin": 717,
          "ymin": 369,
          "xmax": 1383,
          "ymax": 755
        },
        "padding": {
          "horizontalRatioPerSide": 0.2,
          "verticalRatioPerSide": 0.1
        },
        "extendCropCanvas": true,
        "outputSize": {
          "width": 666,
          "height": 386
        }
      },
      "cropImage": "/abs/path/output/crops/pond__x812_y401_x1288_y723.jpg",
      "segmentation": {
        "status": "ok",
        "score": 0.98,
        "cutoutFeatherRadius": 2.5,
        "cutoutEdgeDecontaminated": true,
        "maskBoundingBox": {
          "xmin": 808,
          "ymin": 409,
          "xmax": 1292,
          "ymax": 727
        },
        "foregroundAreaPixels": 63646,
        "backgroundColor": {
          "status": "ok",
          "ringWidth": 16,
          "sampledPixelCount": 7812,
          "meanRgb": {
            "r": 39,
            "g": 107,
            "b": 181
          },
          "medianRgb": {
            "r": 36,
            "g": 104,
            "b": 176
          },
          "recommendedCanvasColor": {
            "r": 39,
            "g": 107,
            "b": 181
          }
        },
        "maskImage": "/abs/path/output/masks/pond__x812_y401_x1288_y723__mask.png",
        "cutoutImage": "/abs/path/output/cutouts/pond__x812_y401_x1288_y723__cutout.png",
        "backgroundCropImage": "/abs/path/output/background-crops/pond__x812_y401_x1288_y723__background-crop.jpg",
        "backgroundCropEdgeSoftness": 3.5,
        "backgroundCrop": {
          "status": "ok",
          "style": "gradient",
          "fillColorSource": "recommendedCanvasColor",
          "fallbackColor": {
            "r": 39,
            "g": 107,
            "b": 181
          },
          "gradientGrid": 5,
          "gradientBlurRadius": 56,
          "gradientSampleCount": 4120,
          "requestedPaddingPreserved": true,
          "outputSize": {
            "width": 666,
            "height": 386
          }
        }
      }
    }
  ]
}
```

## Progress Output

The detector now prints live progress to stderr while running.

Typical output looks like:

```text
Loading model IDEA-Research/grounding-dino-base on mps (cache: /.../.model-cache, offline: False)...
[1/6] IMG_6004.jpg processing /path/to/IMG_6004.jpg...
[1/6] IMG_6004.jpg mode: fast
[1/6] IMG_6004.jpg variant 1/1: original with 1 tile(s)
[1/6] IMG_6004.jpg complete: 2 detection(s)
```

## Notes

- The first online run may take time because the model has to be downloaded into the local cache.
- If `--segment` is enabled, the first segmentation run also downloads the SAM model into the same cache.
- `--device mps` is usually the right choice on Apple Silicon.
- `--mode auto` is usually the best default.
- If JPG batches still feel too slow, force `--mode fast`.
- If HEIC glare and surface reflections are a problem, try `--mode balanced` or `--mode robust`.
- Multi-fish photos are supported. Each detected fish gets its own bounding box, crop, and JSON entry.
- Segmentation is optional and does not replace box detection. If masks are unreliable for a photo, the rectangular crop and bounding box still remain available.
- Feathered cutouts and edge decontamination are optional feature flags. Leave them off if you want fully literal mask edges.
- The flattened background crop is meant for publishing workflows where a transparent cutout would otherwise reveal edge bleed against the wrong canvas color.
- If the flattened background still looks too flat, enable `--background-crop-gradient`.
- If the fish edge looks too sharp in the flattened crop, increase `--background-crop-edge-softness`.
