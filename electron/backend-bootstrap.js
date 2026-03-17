const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const MANAGED_PYTHON_VERSION = process.env.VOICE_FACTORY_MANAGED_PYTHON_VERSION || "3.11";
const BOOTSTRAP_SCHEMA_VERSION = 2;
const NVIDIA_TORCH_INDEX_URL =
  process.env.VOICE_FACTORY_TORCH_INDEX_URL || "https://download.pytorch.org/whl/cu130";
const COMMON_PACKAGES = [
  "fastapi>=0.115.0",
  "gradio>=5.21.0",
  "httpx>=0.28.0",
  "huggingface_hub>=0.30.0",
  "kizuna-voice-designer[gguf] @ git+https://github.com/kizuna-intelligence/kizuna-voice-designer.git",
  "librosa==0.10.2",
  "numpy>=1.26.0",
  "onnxruntime>=1.17.0",
  "pydantic>=2.10.0",
  "qwen-tts>=0.1.1",
  "soundfile>=0.13.0",
  "tqdm>=4.66.0",
  "transformers>=4.49.0",
  "uvicorn>=0.34.0",
  "piper-train @ https://github.com/ayutaz/piper-plus/archive/refs/heads/dev.zip#subdirectory=src/python",
];

function managedPaths(app, backendProfile) {
  const userDataRoot = app.getPath("userData");
  const stateRoot = path.join(userDataRoot, "managed-backend", backendProfile);
  const uvInstallDir = path.join(stateRoot, "tools", "uv");
  const uvCacheDir = path.join(stateRoot, "tools", "uv-cache");
  const pythonInstallDir = path.join(stateRoot, "python");
  const runtimeDir = path.join(stateRoot, "runtime");
  const venvDir = path.join(runtimeDir, "venv");
  const workspaceDir = path.join(stateRoot, "workspace");
  const cacheDir = path.join(stateRoot, "cache");
  const bootstrapDir = path.join(stateRoot, "bootstrap");
  const metaPath = path.join(bootstrapDir, "backend-meta.json");
  const uvBinary =
    process.platform === "win32" ? path.join(uvInstallDir, "uv.exe") : path.join(uvInstallDir, "uv");
  const pythonBinary =
    process.platform === "win32"
      ? path.join(venvDir, "Scripts", "python.exe")
      : path.join(venvDir, "bin", "python");
  return {
    userDataRoot,
    stateRoot,
    uvInstallDir,
    uvCacheDir,
    pythonInstallDir,
    runtimeDir,
    venvDir,
    workspaceDir,
    cacheDir,
    bootstrapDir,
    metaPath,
    uvBinary,
    pythonBinary,
  };
}

function ensureDir(targetPath) {
  fs.mkdirSync(targetPath, { recursive: true });
}

function commandExists(command) {
  const locator = process.platform === "win32" ? "where" : "which";
  const result = spawnSync(locator, [command], { stdio: "ignore" });
  return result.status === 0;
}

function runChecked(command, args, options = {}) {
  const result = spawnSync(command, args, {
    ...options,
    encoding: "utf-8",
  });
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `${command} exited with status ${result.status}`);
  }
  return result;
}

function downloadFile(url, destination) {
  return new Promise((resolve, reject) => {
    ensureDir(path.dirname(destination));
    const handleResponse = (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        response.resume();
        downloadFile(response.headers.location, destination).then(resolve, reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download ${url}: HTTP ${response.statusCode}`));
        response.resume();
        return;
      }
      const file = fs.createWriteStream(destination);
      response.pipe(file);
      file.on("finish", () => file.close(resolve));
      file.on("error", reject);
    };

    https
      .get(url, handleResponse)
      .on("error", reject);
  });
}

async function installManagedUv(paths) {
  ensureDir(paths.uvInstallDir);
  const installerPath = path.join(
    os.tmpdir(),
    `voice-factory-uv-installer-${Date.now()}${process.platform === "win32" ? ".ps1" : ".sh"}`
  );
  const installerUrl =
    process.platform === "win32" ? "https://astral.sh/uv/install.ps1" : "https://astral.sh/uv/install.sh";
  await downloadFile(installerUrl, installerPath);
  const env = {
    ...process.env,
    UV_UNMANAGED_INSTALL: paths.uvInstallDir,
    UV_NO_MODIFY_PATH: "1",
  };
  try {
    if (process.platform === "win32") {
      runChecked("powershell", ["-ExecutionPolicy", "Bypass", "-File", installerPath], { env });
    } else {
      runChecked("/bin/sh", [installerPath], { env });
    }
  } finally {
    fs.rmSync(installerPath, { force: true });
  }
  if (!fs.existsSync(paths.uvBinary)) {
    throw new Error(`uv bootstrap failed. Expected executable at ${paths.uvBinary}`);
  }
}

async function ensureUv(paths) {
  if (process.env.VOICE_FACTORY_BOOTSTRAP_UV && fs.existsSync(process.env.VOICE_FACTORY_BOOTSTRAP_UV)) {
    return process.env.VOICE_FACTORY_BOOTSTRAP_UV;
  }
  if (fs.existsSync(paths.uvBinary)) {
    return paths.uvBinary;
  }
  if (commandExists("uv")) {
    return "uv";
  }
  await installManagedUv(paths);
  return paths.uvBinary;
}

function installMeta(appVersion, buildFlavor, backendProfile, cudaChannel) {
  return {
    schemaVersion: BOOTSTRAP_SCHEMA_VERSION,
    appVersion,
    buildFlavor,
    backendProfile,
    cudaChannel,
    pythonVersion: MANAGED_PYTHON_VERSION,
  };
}

function isBootstrapFresh(paths, expectedMeta) {
  if (!fs.existsSync(paths.pythonBinary) || !fs.existsSync(paths.metaPath)) {
    return false;
  }
  try {
    const currentMeta = JSON.parse(fs.readFileSync(paths.metaPath, "utf-8"));
    return Object.keys(expectedMeta).every((key) => currentMeta[key] === expectedMeta[key]);
  } catch (error) {
    return false;
  }
}

function backendEnv(paths, backendRoot, backendProfile, defaultComputeTarget, cudaChannel) {
  const env = {
    PYTHONPATH: path.join(backendRoot, "src"),
    VOICE_FACTORY_WORKSPACE_ROOT: paths.workspaceDir,
    VOICE_FACTORY_BACKEND_PROFILE: backendProfile,
    VOICE_FACTORY_DEFAULT_COMPUTE_TARGET: defaultComputeTarget,
    VOICE_FACTORY_CUDA_CHANNEL: cudaChannel,
    HF_HOME: path.join(paths.cacheDir, "huggingface"),
    TORCH_HOME: path.join(paths.cacheDir, "torch"),
    XDG_CACHE_HOME: paths.cacheDir,
    WANDB_MODE: "disabled",
    WANDB_DISABLED: "true",
    WANDB_SILENT: "true",
  };
  return env;
}

function installPythonEnvironment(uvCommand, paths, backendRoot, backendProfile, cudaChannel) {
  ensureDir(paths.bootstrapDir);
  ensureDir(paths.workspaceDir);
  ensureDir(paths.cacheDir);
  const uvEnv = {
    ...process.env,
    UV_CACHE_DIR: paths.uvCacheDir,
    UV_PYTHON_INSTALL_DIR: paths.pythonInstallDir,
  };
  runChecked(uvCommand, ["venv", paths.venvDir, "--python", MANAGED_PYTHON_VERSION], { env: uvEnv });

  if (backendProfile === "nvidia") {
    runChecked(
      uvCommand,
      [
        "pip",
        "install",
        "--python",
        paths.pythonBinary,
        "--index-url",
        NVIDIA_TORCH_INDEX_URL,
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
      ],
      { env: uvEnv, cwd: backendRoot }
    );
    runChecked(
      uvCommand,
      ["pip", "install", "--python", paths.pythonBinary, "bitsandbytes>=0.45.0"],
      { env: uvEnv, cwd: backendRoot }
    );
  } else {
    runChecked(
      uvCommand,
      ["pip", "install", "--python", paths.pythonBinary, "torch>=2.4.0"],
      { env: uvEnv, cwd: backendRoot }
    );
  }

  runChecked(
    uvCommand,
    ["pip", "install", "--python", paths.pythonBinary, ...COMMON_PACKAGES],
    { env: uvEnv, cwd: backendRoot }
  );
  runChecked(
    uvCommand,
    ["pip", "install", "--python", paths.pythonBinary, "--no-deps", "--force-reinstall", "--editable", backendRoot],
    { env: uvEnv, cwd: backendRoot }
  );
}

async function ensureManagedBackend({
  app,
  backendRoot,
  backendProfile,
  buildFlavor,
  defaultComputeTarget,
  cudaChannel,
  appVersion,
}) {
  const paths = managedPaths(app, backendProfile);
  const expectedMeta = installMeta(appVersion, buildFlavor, backendProfile, cudaChannel);
  if (!isBootstrapFresh(paths, expectedMeta)) {
    const uvCommand = await ensureUv(paths);
    installPythonEnvironment(uvCommand, paths, backendRoot, backendProfile, cudaChannel);
    ensureDir(paths.bootstrapDir);
    fs.writeFileSync(paths.metaPath, JSON.stringify(expectedMeta, null, 2), "utf-8");
  }
  return {
    pythonPath: paths.pythonBinary,
    env: backendEnv(paths, backendRoot, backendProfile, defaultComputeTarget, cudaChannel),
    paths,
  };
}

module.exports = {
  ensureManagedBackend,
};
