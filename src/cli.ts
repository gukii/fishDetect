import { existsSync } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

type Device = "auto" | "cpu" | "cuda" | "mps";
type DetectionMode = "auto" | "fast" | "balanced" | "robust";

type CliOptions = {
  input: string;
  output: string;
  pythonBin: string;
  modelId: string;
  segment: boolean;
  segmentExportBackgroundCrop: boolean;
  backgroundCropGradient: boolean;
  backgroundCropGradientGrid: number;
  backgroundCropGradientBlur: number;
  backgroundCropEdgeSoftness?: number;
  segmentationModelId: string;
  segmentationMinScore: number;
  cutoutFeatherRadius: number;
  cutoutStoreBackgroundColor: boolean;
  cutoutBackgroundRing: number;
  cutoutDecontaminateEdges: boolean;
  modelCacheDir: string;
  offline: boolean;
  mode: DetectionMode;
  boxThreshold: number;
  textThreshold: number;
  maxBoxAreaRatio: number;
  cropPadX: number;
  cropPadY: number;
  extendCropCanvas: boolean;
  extensions?: string[];
  limit?: number;
  device: Device;
};

type DetectorManifest = {
  manifestPath: string;
  imagesProcessed: number;
  detectionsWritten: number;
  cropsWritten: number;
  backgroundCropsWritten?: number;
  modelId: string;
};

const projectRoot = fileURLToPath(new URL("..", import.meta.url));

function printHelp(): void {
  console.log(`Usage:
  pnpm detect --input <path> [--output ./output] [--python-bin ./.venv/bin/python]

Options:
  --input           Photo file or directory to scan
  --output          Output root for JSON and crop files (default: ./output)
  --python-bin      Python executable to use (default: ./.venv/bin/python, then python3.12)
  --model           Hugging Face model id (default: IDEA-Research/grounding-dino-base)
  --segment         Generate fish masks and transparent cutouts in addition to box crops
  --segment-export-background-crop Export a padded segmentation crop flattened onto sampled background color
  --background-crop-gradient Use a filtered multi-point gradient instead of one flat background color
  --background-crop-gradient-grid Gradient anchor grid size (default: 4)
  --background-crop-gradient-blur Blur radius used to smooth the generated gradient (default: 48)
  --background-crop-edge-softness Feather radius used when compositing the segmented fish onto the background crop
  --segmentation-model Model id or local directory for segmentation (default: facebook/sam-vit-base)
  --segmentation-min-score Minimum SAM mask score to keep (default: 0.88)
  --cutout-feather-radius Feather cutout alpha edges by this many pixels (default: 0)
  --cutout-store-background-color Sample and store a recommended background color around each mask
  --cutout-background-ring Ring width in pixels used for background color sampling (default: 12)
  --cutout-decontaminate-edges Remove background color spill from semi-transparent cutout edges
  --model-cache-dir Local cache directory for model files (default: ./.model-cache)
  --offline         Require model files to be loaded from local cache only
  --mode            auto | fast | balanced | robust (default: auto)
  --box-threshold   Detection confidence threshold (default: 0.25)
  --text-threshold  Text matching threshold when supported (default: 0.20)
  --max-box-area-ratio Reject boxes larger than this fraction of the image (default: 0.35)
  --crop-pad-x      Extra horizontal crop space per side as a fraction of box width (default: 0)
  --crop-pad-y      Extra vertical crop space per side as a fraction of box height (default: 0)
  --extend-crop-canvas Extend crop beyond image edges using replicated edge pixels
  --extensions      Comma-separated file extensions to include, for example .heic,.heif
  --limit           Maximum number of images to process
  --device          auto | cpu | cuda | mps (default: auto)
  --help            Show this message`);
}

function takeValue(argv: string[], index: number, flag: string): string {
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

function parseNumber(raw: string, flag: string): number {
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid numeric value for ${flag}: ${raw}`);
  }
  return value;
}

function parsePositiveInteger(raw: string, flag: string): number {
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Invalid positive integer for ${flag}: ${raw}`);
  }
  return value;
}

function parseExtensions(raw: string): string[] {
  const extensions = raw
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean)
    .map((part) => (part.startsWith(".") ? part : `.${part}`));

  if (extensions.length === 0) {
    throw new Error("At least one extension must be provided.");
  }

  return extensions;
}

function resolveDefaultPython(): string {
  const inRepo = path.join(projectRoot, ".venv", "bin", "python");
  if (existsSync(inRepo)) {
    return inRepo;
  }
  return "python3.12";
}

function normalizeExecutable(value: string): string {
  if (path.isAbsolute(value) || value.startsWith(".") || value.includes(path.sep)) {
    return path.resolve(process.cwd(), value);
  }
  return value;
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    input: "",
    output: path.resolve(projectRoot, "output"),
    pythonBin: process.env.KOI_DETECTOR_PYTHON ?? resolveDefaultPython(),
    modelId: "IDEA-Research/grounding-dino-base",
    segment: false,
    segmentExportBackgroundCrop: false,
    backgroundCropGradient: false,
    backgroundCropGradientGrid: 4,
    backgroundCropGradientBlur: 48,
    segmentationModelId: "facebook/sam-vit-base",
    segmentationMinScore: 0.88,
    cutoutFeatherRadius: 0,
    cutoutStoreBackgroundColor: false,
    cutoutBackgroundRing: 12,
    cutoutDecontaminateEdges: false,
    modelCacheDir: path.resolve(projectRoot, ".model-cache"),
    offline: false,
    mode: "auto",
    boxThreshold: 0.25,
    textThreshold: 0.2,
    maxBoxAreaRatio: 0.35,
    cropPadX: 0,
    cropPadY: 0,
    extendCropCanvas: false,
    device: "auto"
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--input":
        options.input = path.resolve(process.cwd(), takeValue(argv, index, arg));
        index += 1;
        break;
      case "--output":
        options.output = path.resolve(process.cwd(), takeValue(argv, index, arg));
        index += 1;
        break;
      case "--python-bin":
        options.pythonBin = normalizeExecutable(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--model":
        options.modelId = takeValue(argv, index, arg);
        index += 1;
        break;
      case "--segment":
        options.segment = true;
        break;
      case "--segment-export-background-crop":
        options.segmentExportBackgroundCrop = true;
        break;
      case "--background-crop-gradient":
        options.backgroundCropGradient = true;
        break;
      case "--background-crop-gradient-grid":
        options.backgroundCropGradientGrid = parsePositiveInteger(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--background-crop-gradient-blur":
        options.backgroundCropGradientBlur = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--background-crop-edge-softness":
        options.backgroundCropEdgeSoftness = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--segmentation-model":
        options.segmentationModelId = takeValue(argv, index, arg);
        index += 1;
        break;
      case "--segmentation-min-score":
        options.segmentationMinScore = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--cutout-feather-radius":
        options.cutoutFeatherRadius = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--cutout-store-background-color":
        options.cutoutStoreBackgroundColor = true;
        break;
      case "--cutout-background-ring":
        options.cutoutBackgroundRing = parsePositiveInteger(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--cutout-decontaminate-edges":
        options.cutoutDecontaminateEdges = true;
        break;
      case "--model-cache-dir":
        options.modelCacheDir = path.resolve(process.cwd(), takeValue(argv, index, arg));
        index += 1;
        break;
      case "--offline":
        options.offline = true;
        break;
      case "--mode": {
        const value = takeValue(argv, index, arg) as DetectionMode;
        if (!["auto", "fast", "balanced", "robust"].includes(value)) {
          throw new Error(`Invalid mode: ${value}`);
        }
        options.mode = value;
        index += 1;
        break;
      }
      case "--box-threshold":
        options.boxThreshold = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--text-threshold":
        options.textThreshold = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--device": {
        const value = takeValue(argv, index, arg) as Device;
        if (!["auto", "cpu", "cuda", "mps"].includes(value)) {
          throw new Error(`Invalid device: ${value}`);
        }
        options.device = value;
        index += 1;
        break;
      }
      case "--max-box-area-ratio":
        options.maxBoxAreaRatio = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--crop-pad-x":
        options.cropPadX = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--crop-pad-y":
        options.cropPadY = parseNumber(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--extend-crop-canvas":
        options.extendCropCanvas = true;
        break;
      case "--extensions":
        options.extensions = parseExtensions(takeValue(argv, index, arg));
        index += 1;
        break;
      case "--limit":
        options.limit = parsePositiveInteger(takeValue(argv, index, arg), arg);
        index += 1;
        break;
      case "--help":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!options.input) {
    throw new Error("--input is required");
  }

  return options;
}

async function runDetector(options: CliOptions): Promise<DetectorManifest> {
  const detectorScript = path.join(projectRoot, "python", "detect_koi.py");
  if (!existsSync(options.input)) {
    throw new Error(`Input path does not exist or is not accessible: ${options.input}`);
  }
  await mkdir(options.output, { recursive: true });
  await mkdir(options.modelCacheDir, { recursive: true });
  const detectorArgs = [
    detectorScript,
    "--input",
    options.input,
    "--output-root",
    options.output,
    "--model-id",
    options.modelId,
    "--segmentation-model-id",
    options.segmentationModelId,
    "--segmentation-min-score",
    String(options.segmentationMinScore),
    "--background-crop-gradient-grid",
    String(options.backgroundCropGradientGrid),
    "--background-crop-gradient-blur",
    String(options.backgroundCropGradientBlur),
    "--cutout-feather-radius",
    String(options.cutoutFeatherRadius),
    "--cutout-background-ring",
    String(options.cutoutBackgroundRing),
    "--model-cache-dir",
    options.modelCacheDir,
    "--mode",
    options.mode,
    "--box-threshold",
    String(options.boxThreshold),
    "--text-threshold",
    String(options.textThreshold),
    "--max-box-area-ratio",
    String(options.maxBoxAreaRatio),
    "--crop-pad-x",
    String(options.cropPadX),
    "--crop-pad-y",
    String(options.cropPadY),
    "--device",
    options.device
  ];

  if (options.limit !== undefined) {
    detectorArgs.push("--limit", String(options.limit));
  }
  if (options.extensions !== undefined) {
    detectorArgs.push("--extensions", options.extensions.join(","));
  }
  if (options.extendCropCanvas) {
    detectorArgs.push("--extend-crop-canvas");
  }
  if (options.offline) {
    detectorArgs.push("--offline");
  }
  if (options.segment) {
    detectorArgs.push("--segment");
  }
  if (options.segmentExportBackgroundCrop) {
    detectorArgs.push("--segment-export-background-crop");
  }
  if (options.backgroundCropGradient) {
    detectorArgs.push("--background-crop-gradient");
  }
  if (options.backgroundCropEdgeSoftness !== undefined) {
    detectorArgs.push("--background-crop-edge-softness", String(options.backgroundCropEdgeSoftness));
  }
  if (options.cutoutStoreBackgroundColor) {
    detectorArgs.push("--cutout-store-background-color");
  }
  if (options.cutoutDecontaminateEdges) {
    detectorArgs.push("--cutout-decontaminate-edges");
  }

  return await new Promise<DetectorManifest>((resolve, reject) => {
    const child = spawn(
      options.pythonBin,
      detectorArgs,
      {
        cwd: projectRoot,
        env: {
          ...process.env,
          HF_HUB_DISABLE_XET: "1"
        },
        stdio: ["ignore", "pipe", "pipe"]
      }
    );

    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      process.stderr.write(chunk);
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            stderr.trim() || `Detector exited with code ${code ?? "unknown"}`
          )
        );
        return;
      }

      try {
        const parsed = JSON.parse(stdout.trim()) as DetectorManifest;
        resolve(parsed);
      } catch (error) {
        reject(
          new Error(
            `Unable to parse detector output as JSON.\n${String(error)}\nRaw stdout:\n${stdout}\nRaw stderr:\n${stderr}`
          )
        );
      }
    });
  });
}

async function printSampleOutput(manifestPath: string): Promise<void> {
  const raw = await readFile(manifestPath, "utf8");
  const parsed = JSON.parse(raw) as { images: Array<{ detections: unknown[] }> };
  console.log(`Manifest: ${manifestPath}`);
  console.log(`Images in manifest: ${parsed.images.length}`);
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const result = await runDetector(options);
  console.log(
    (
      [
        `Processed ${result.imagesProcessed} image(s)`,
        `wrote ${result.detectionsWritten} detection(s)`,
        `and ${result.cropsWritten} crop file(s).`,
        result.backgroundCropsWritten !== undefined
          ? `Background crops: ${result.backgroundCropsWritten}.`
          : "",
        `Manifest: ${result.manifestPath}`
      ]
        .filter(Boolean)
        .join(" ")
    )
  );
  await printSampleOutput(result.manifestPath);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  if (
    message.includes("Missing Python dependencies") ||
    message.includes("No such file or directory") ||
    message.includes("spawn")
  ) {
    console.error(
      "If Python dependencies are missing, create a venv with `python3.12 -m venv .venv` and install `python/requirements.txt`."
    );
  }
  process.exit(1);
});
