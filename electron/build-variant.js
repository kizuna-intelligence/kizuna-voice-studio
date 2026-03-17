const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const { builderConfig, variantConfig, variants } = require("./variant-manifest");

function usage() {
  const keys = Object.keys(variants).join(", ");
  console.error(`Usage: node build-variant.js <variant>\nAvailable variants: ${keys}`);
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
