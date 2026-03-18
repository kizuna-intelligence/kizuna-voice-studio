const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const { builderConfig, variantConfig, variants } = require("./variant-manifest");

const WINDOWS_BUNDLED_WHEELS = ["jieba-fast==0.53"];

function usage() {
  const keys = Object.keys(variants).join(", ");
  console.error(`Usage: node build-variant.js <variant>\nAvailable variants: ${keys}`);
}

function runChecked(command, args, options = {}) {
  const result = spawnSync(command, args, {
    ...options,
    encoding: "utf-8",
    stdio: "inherit",
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${command} exited with status ${result.status}`);
  }
}

function canRun(command, args = []) {
  const result = spawnSync(command, args, {
    encoding: "utf-8",
    stdio: "ignore",
  });
  return result.status === 0;
}

function resolveWindowsBuildPython() {
  if (process.env.VOICE_FACTORY_BUILD_PYTHON) {
    return [process.env.VOICE_FACTORY_BUILD_PYTHON, []];
  }
  if (canRun("py", ["-3.11", "-c", "import sys; print(sys.version)"])) {
    return ["py", ["-3.11"]];
  }
  if (canRun("python", ["-c", "import sys; print(sys.version)"])) {
    return ["python", []];
  }
  throw new Error(
    "Windows build requires Python 3.11 to prebuild bundled wheels. Set VOICE_FACTORY_BUILD_PYTHON if needed."
  );
}

function prepareWindowsWheelhouse(variant) {
  if (process.platform !== "win32" || variant.platform !== "win") {
    return;
  }

  const [pythonCommand, pythonPrefixArgs] = resolveWindowsBuildPython();
  const wheelhouseDir = path.join(__dirname, "wheelhouse", variant.key);
  fs.rmSync(wheelhouseDir, { recursive: true, force: true });
  fs.mkdirSync(wheelhouseDir, { recursive: true });

  for (const requirement of WINDOWS_BUNDLED_WHEELS) {
    runChecked(
      pythonCommand,
      [
        ...pythonPrefixArgs,
        "-m",
        "pip",
        "wheel",
        "--wheel-dir",
        wheelhouseDir,
        requirement,
      ],
      {
        env: {
          ...process.env,
          PIP_DISABLE_PIP_VERSION_CHECK: "1",
        },
      }
    );
  }
}

const variantKey = process.argv[2];
if (!variantKey) {
  usage();
  process.exit(1);
}

let variant;
try {
  variant = variantConfig(variantKey);
} catch (error) {
  console.error(String(error));
  usage();
  process.exit(1);
}

prepareWindowsWheelhouse(variant);

const configPath = path.join(os.tmpdir(), `voice-factory-builder-${variant.key}.json`);
fs.writeFileSync(configPath, JSON.stringify(builderConfig(variant), null, 2));

const platformFlag =
  variant.platform === "win" ? "--win" : variant.platform === "linux" ? "--linux" : "--mac";
const archFlag = variant.arch === "arm64" ? "--arm64" : "--x64";
const electronBuilderCli = require.resolve("electron-builder/cli.js");

const result = spawnSync(
  process.execPath,
  [electronBuilderCli, platformFlag, archFlag, "--config", configPath],
  {
    stdio: "inherit",
    env: {
      ...process.env,
      VOICE_FACTORY_DISTRIBUTION_FLAVOR: variant.key,
      VOICE_FACTORY_BACKEND_PROFILE: variant.backendProfile,
      VOICE_FACTORY_DEFAULT_COMPUTE_TARGET: variant.defaultComputeTarget,
      VOICE_FACTORY_CUDA_CHANNEL: variant.cudaChannel || "none",
    },
  }
);

if (result.error) {
  throw result.error;
}
process.exit(result.status ?? 0);
