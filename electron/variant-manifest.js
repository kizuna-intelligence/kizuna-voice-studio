const path = require("path");

const commonFiles = [
  "backend-bootstrap.js",
  "main.js",
  "preload.js",
  "index.html",
  "renderer.js",
  "styles.css",
  "build-variant.js",
  "variant-manifest.js",
];

const backendResources = [
  {
    from: path.join("..", "src"),
    to: path.join("backend", "src"),
    filter: ["**/*"],
  },
  {
    from: path.join("..", "examples"),
    to: path.join("backend", "examples"),
    filter: ["**/*"],
  },
  {
    from: path.join("..", "pyproject.toml"),
    to: path.join("backend", "pyproject.toml"),
  },
  {
    from: path.join("..", "README.md"),
    to: path.join("backend", "README.md"),
  },
];

function variantResources(variant) {
  const resources = [...backendResources];
  const wheelhouseDir = path.join("wheelhouse", variant.key);
  if (variant.platform === "win") {
    resources.push({
      from: wheelhouseDir,
      to: path.join("backend", "wheelhouse"),
      filter: ["**/*"],
    });
  }
  return resources;
}

const variants = {
  "windows-nvidia": {
    key: "windows-nvidia",
    platform: "win",
    arch: "x64",
    productName: "Voice Factory Windows NVIDIA",
    artifactName: "VoiceFactory-Windows-NVIDIA-${version}.${ext}",
    appId: "dev.mugen.voicefactory.windows.nvidia",
    backendProfile: "nvidia",
    defaultComputeTarget: "gpu:0",
    cudaChannel: "latest",
    description: "Windows build for recent NVIDIA CUDA environments.",
  },
  "windows-amd": {
    key: "windows-amd",
    platform: "win",
    arch: "x64",
    productName: "Voice Factory Windows AMD",
    artifactName: "VoiceFactory-Windows-AMD-${version}.${ext}",
    appId: "dev.mugen.voicefactory.windows.amd",
    backendProfile: "amd",
    defaultComputeTarget: "cpu",
    cudaChannel: "none",
    description: "Windows build for AMD and non-NVIDIA machines using the compatibility path.",
  },
  "linux-nvidia": {
    key: "linux-nvidia",
    platform: "linux",
    arch: "x64",
    productName: "Voice Factory Linux NVIDIA",
    artifactName: "VoiceFactory-Linux-NVIDIA-${version}.${ext}",
    appId: "dev.mugen.voicefactory.linux.nvidia",
    backendProfile: "nvidia",
    defaultComputeTarget: "gpu:0",
    cudaChannel: "latest",
    description: "Linux build for recent NVIDIA CUDA environments.",
  },
  "macos-apple-silicon": {
    key: "macos-apple-silicon",
    platform: "mac",
    arch: "arm64",
    productName: "Voice Factory macOS Apple Silicon",
    artifactName: "VoiceFactory-macOS-AppleSilicon-${version}.${ext}",
    appId: "dev.mugen.voicefactory.macos.apple",
    backendProfile: "apple",
    defaultComputeTarget: "auto",
    description: "macOS build for Apple Silicon machines.",
  },
};

function variantConfig(key) {
  const variant = variants[key];
  if (!variant) {
    throw new Error(`Unknown Electron build variant: ${key}`);
  }
  return variant;
}

function builderConfig(variant) {
  return {
    appId: variant.appId,
    productName: variant.productName,
    artifactName: variant.artifactName,
    directories: {
      output: path.join("dist", variant.key),
    },
    files: commonFiles,
    extraMetadata: {
      voiceFactoryDistributionFlavor: variant.key,
      voiceFactoryBackendProfile: variant.backendProfile,
      voiceFactoryDefaultComputeTarget: variant.defaultComputeTarget,
      voiceFactoryCudaChannel: variant.cudaChannel || "none",
    },
    extraResources: variantResources(variant),
    mac: {
      target: ["dmg", "zip"],
      artifactName: variant.artifactName,
    },
    win: {
      target: ["nsis", "zip"],
      artifactName: variant.artifactName,
    },
    linux: {
      target: ["AppImage", "tar.gz"],
      artifactName: variant.artifactName,
    },
  };
}

module.exports = {
  variants,
  variantConfig,
  builderConfig,
};
