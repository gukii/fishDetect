import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";

type DemoOptions = {
  input: string;
  outputRoot: string;
  publicJsonPath: string;
  device: "auto" | "cpu" | "cuda" | "mps";
};

type DetectionRecord = {
  label: string;
  matchedPrompt?: string;
  confidence: number;
  boundingBox: Record<string, number>;
  cropRegion?: unknown;
  cropImage?: string;
  sourcePass?: unknown;
  segmentation?: Record<string, unknown>;
};

type ImagePayload = {
  sourceImage: string;
  relativeSourcePath: string;
  imageSize: Record<string, number>;
  detector: Record<string, unknown>;
  detections: DetectionRecord[];
};

type ImageJson = ImagePayload;

function takeValue(argv: string[], index: number, flag: string): string {
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

function printHelp(): void {
  console.log(`Usage:
  pnpm docs:example -- --input /absolute/path/to/IMG_6008.jpg

Options:
  --input         Single source image file to use for docs examples
  --output        Private generated docs output root (default: ./docs/generated/<image-stem>)
  --public-json   Public sanitized demo JSON output path (default: ./docs/public-assets/demo-json/<image-stem>.demo.json)
  --device        auto | cpu | cuda | mps (default: mps)
  --help          Show this message`);
}

function parseArgs(argv: string[]): DemoOptions {
  let input = "";
  let outputRoot = "";
  let publicJsonPath = "";
  let device: DemoOptions["device"] = "mps";

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--") {
      continue;
    }
    switch (arg) {
      case "--input":
        input = path.resolve(process.cwd(), takeValue(argv, index, arg));
        index += 1;
        break;
      case "--output":
        outputRoot = path.resolve(process.cwd(), takeValue(argv, index, arg));
        index += 1;
        break;
      case "--public-json":
        publicJsonPath = path.resolve(process.cwd(), takeValue(argv, index, arg));
        index += 1;
        break;
      case "--device": {
        const value = takeValue(argv, index, arg) as DemoOptions["device"];
        if (!["auto", "cpu", "cuda", "mps"].includes(value)) {
          throw new Error(`Invalid device: ${value}`);
        }
        device = value;
        index += 1;
        break;
      }
      case "--help":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!input) {
    throw new Error("--input is required");
  }
  if (!existsSync(input)) {
    throw new Error(`Input path does not exist: ${input}`);
  }
  const extension = path.extname(input).toLowerCase();
  const supported = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"]);
  if (!supported.has(extension)) {
    throw new Error(`Unsupported input file extension for docs example: ${extension}`);
  }

  const imageStem = path.basename(input, extension);
  if (!outputRoot) {
    outputRoot = path.resolve(process.cwd(), "docs", "generated", imageStem);
  }
  if (!publicJsonPath) {
    publicJsonPath = path.resolve(process.cwd(), "docs", "public-assets", "demo-json", `${imageStem}.demo.json`);
  }

  return { input, outputRoot, publicJsonPath, device };
}

function relativeArtifact(filePath: string | undefined): string | null {
  if (!filePath) {
    return null;
  }
  return path.join("docs", "generated", path.basename(path.dirname(path.dirname(filePath))), path.basename(path.dirname(filePath)), path.basename(filePath));
}

function sanitizeSegmentation(segmentation: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  if (!segmentation) {
    return undefined;
  }
  const artifactFiles = {
    maskImage: relativeArtifact(typeof segmentation.maskImage === "string" ? segmentation.maskImage : undefined),
    cutoutImage: relativeArtifact(typeof segmentation.cutoutImage === "string" ? segmentation.cutoutImage : undefined),
    backgroundCropImage: relativeArtifact(
      typeof segmentation.backgroundCropImage === "string" ? segmentation.backgroundCropImage : undefined
    )
  };

  return {
    status: segmentation.status,
    score: segmentation.score,
    maskBoundingBox: segmentation.maskBoundingBox,
    foregroundAreaPixels: segmentation.foregroundAreaPixels,
    cutoutFeatherRadius: segmentation.cutoutFeatherRadius,
    cutoutEdgeDecontaminated: segmentation.cutoutEdgeDecontaminated,
    backgroundCropEdgeSoftness: segmentation.backgroundCropEdgeSoftness,
    backgroundColor: segmentation.backgroundColor,
    backgroundCrop: segmentation.backgroundCrop,
    artifactFiles
  };
}

function sanitizeImageJson(imageJson: ImageJson, outputRoot: string): Record<string, unknown> {
  return {
    demoImage: path.basename(imageJson.relativeSourcePath),
    note: "Original photo is private and not committed. This JSON is a sanitized example.",
    imageSize: imageJson.imageSize,
    detector: {
      modelId: imageJson.detector.modelId,
      segmentationEnabled: imageJson.detector.segmentationEnabled,
      segmentExportBackgroundCrop: imageJson.detector.segmentExportBackgroundCrop,
      backgroundCropGradient: imageJson.detector.backgroundCropGradient,
      backgroundCropGradientGrid: imageJson.detector.backgroundCropGradientGrid,
      backgroundCropGradientBlur: imageJson.detector.backgroundCropGradientBlur,
      backgroundCropEdgeSoftness: imageJson.detector.backgroundCropEdgeSoftness,
      segmentationModelId: imageJson.detector.segmentationModelId,
      segmentationMinScore: imageJson.detector.segmentationMinScore,
      cutoutFeatherRadius: imageJson.detector.cutoutFeatherRadius,
      cutoutBackgroundRing: imageJson.detector.cutoutBackgroundRing,
      cutoutDecontaminateEdges: imageJson.detector.cutoutDecontaminateEdges,
      device: imageJson.detector.device,
      requestedMode: imageJson.detector.requestedMode,
      cropPadX: imageJson.detector.cropPadX,
      cropPadY: imageJson.detector.cropPadY
    },
    generatedOutputRoot: path.relative(process.cwd(), outputRoot),
    detections: imageJson.detections.map((detection) => ({
      label: detection.label,
      matchedPrompt: detection.matchedPrompt,
      confidence: detection.confidence,
      boundingBox: detection.boundingBox,
      cropRegion: detection.cropRegion,
      sourcePass: detection.sourcePass,
      cropImage: relativeArtifact(detection.cropImage),
      segmentation: sanitizeSegmentation(detection.segmentation)
    }))
  };
}

async function runDetect(input: string, outputRoot: string, device: DemoOptions["device"]): Promise<void> {
  await mkdir(outputRoot, { recursive: true });

  const args = [
    "detect",
    "--input",
    input,
    "--output",
    outputRoot,
    "--mode",
    "fast",
    "--device",
    device,
    "--segment",
    "--crop-pad-x",
    "0.20",
    "--crop-pad-y",
    "0.10",
    "--segment-export-background-crop",
    "--background-crop-gradient",
    "--background-crop-gradient-grid",
    "5",
    "--background-crop-gradient-blur",
    "56",
    "--background-crop-edge-softness",
    "3.5",
    "--cutout-background-ring",
    "18",
    "--cutout-decontaminate-edges"
  ];

  await new Promise<void>((resolve, reject) => {
    const child = spawn("pnpm", args, {
      cwd: process.cwd(),
      stdio: "inherit",
      env: { ...process.env, HF_HUB_DISABLE_XET: "1" }
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`docs:example detect step exited with code ${code ?? "unknown"}`));
    });
  });
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const extension = path.extname(options.input).toLowerCase();
  const imageStem = path.basename(options.input, extension);

  await runDetect(options.input, options.outputRoot, options.device);

  const imageJsonPath = path.join(options.outputRoot, "json", `${imageStem}.json`);
  const raw = await readFile(imageJsonPath, "utf8");
  const imageJson = JSON.parse(raw) as ImageJson;
  const sanitized = sanitizeImageJson(imageJson, options.outputRoot);

  await mkdir(path.dirname(options.publicJsonPath), { recursive: true });
  await writeFile(options.publicJsonPath, `${JSON.stringify(sanitized, null, 2)}\n`, "utf8");

  console.log(`Public demo JSON: ${path.relative(process.cwd(), options.publicJsonPath)}`);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
