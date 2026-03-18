const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const MANAGED_PYTHON_VERSION = process.env.VOICE_FACTORY_MANAGED_PYTHON_VERSION || "3.11";
const BOOTSTRAP_SCHEMA_VERSION = 8;
const NVIDIA_TORCH_INDEX_URL =
  process.env.VOICE_FACTORY_TORCH_INDEX_URL || "https://download.pytorch.org/whl/cu130";
const WINDOWS_BUNDLED_WHEEL_REQUIREMENTS = ["jieba-fast==0.53"];
const COMMON_PACKAGES = [
  "fastapi>=0.115.0",
  "gradio>=5.21.0",
  "httpx>=0.28.0",
  "huggingface_hub>=0.30.0",
  "librosa==0.10.2",
  "numpy>=1.26.0",
  "onnxruntime>=1.17.0",
  "pydantic>=2.10.0",
  "pyopenjtalk>=0.4.1",
  "qwen-tts>=0.1.1",
  "soundfile>=0.13.0",
  "tqdm>=4.66.0",
  "transformers>=4.49.0,<5",
  "uvicorn>=0.34.0",
  "piper-train @ https://github.com/ayutaz/piper-plus/archive/refs/heads/dev.zip#subdirectory=src/python",
];
const PROFILE_PACKAGES = {
  default: [
    "kizuna-voice-designer[gguf] @ git+https://github.com/kizuna-intelligence/kizuna-voice-designer.git",
  ],
  amd: [
    "kizuna-voice-designer[gguf] @ git+https://github.com/kizuna-intelligence/kizuna-voice-designer.git",
  ],
};
const PACKAGE_RUNTIME_PACKAGES = {
  piper: [
    "numpy>=1.26.0,<2",
    "onnxruntime>=1.17.0",
    "pyopenjtalk>=0.4.1",
    "piper-train @ https://github.com/ayutaz/piper-plus/archive/refs/heads/dev.zip#subdirectory=src/python",
    "pytorch-lightning==2.6.1",
    "tensorboard==2.20.0",
    "torchmetrics==1.9.0",
  ],
  sbv2: [
    "setuptools>=68,<81",
    "numpy<2",
    "style-bert-vits2[torch] @ git+https://github.com/litagin02/Style-Bert-VITS2.git",
  ],
  miotts: [
    "httpx>=0.28.0",
  ],
};

function managedPaths(app, backendProfile) {
  const userDataRoot = app.getPath("userData");
  const managedStateRoot =
    process.env.VOICE_FACTORY_MANAGED_STATE_ROOT ||
    (process.platform === "win32"
      ? path.join(process.env.LOCALAPPDATA || app.getPath("appData"), "KizunaVoiceStudio")
      : path.join(userDataRoot, "managed-backend"));
  const stateRoot = path.join(managedStateRoot, backendProfile);
  const uvInstallDir = path.join(stateRoot, "tools", "uv");
  const uvCacheDir = path.join(stateRoot, "tools", "uv-cache");
  const pythonInstallDir = path.join(stateRoot, "python");
  const runtimeDir = path.join(stateRoot, "runtime");
  const venvDir = path.join(runtimeDir, "venv");
  const workspaceDir = path.join(stateRoot, "workspace");
  const cacheDir = path.join(stateRoot, "cache");
  const bootstrapDir = path.join(stateRoot, "bootstrap");
  const packageRuntimeDir = path.join(stateRoot, "package-runtimes");
  const metaPath = path.join(bootstrapDir, "backend-meta.json");
  const bundledWheelhouseDir = path.join(backendRootForResources(), "wheelhouse");
  const uvBinary =
    process.platform === "win32" ? path.join(uvInstallDir, "uv.exe") : path.join(uvInstallDir, "uv");
  const pythonBinary =
    process.platform === "win32"
      ? path.join(venvDir, "Scripts", "python.exe")
      : path.join(venvDir, "bin", "python");
  const runtimeVenvDir = (name) => path.join(packageRuntimeDir, name, "venv");
  const runtimePythonBinary = (name) =>
    process.platform === "win32"
      ? path.join(runtimeVenvDir(name), "Scripts", "python.exe")
      : path.join(runtimeVenvDir(name), "bin", "python");
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
    packageRuntimeDir,
    metaPath,
    bundledWheelhouseDir,
    uvBinary,
    pythonBinary,
    packageRuntimePython: {
      piper: runtimePythonBinary("piper"),
      sbv2: runtimePythonBinary("sbv2"),
      miotts: runtimePythonBinary("miotts"),
    },
    packageRuntimeVenvDir: {
      piper: runtimeVenvDir("piper"),
      sbv2: runtimeVenvDir("sbv2"),
      miotts: runtimeVenvDir("miotts"),
    },
  };
}

function backendRootForResources() {
  if (process.resourcesPath && fs.existsSync(path.join(process.resourcesPath, "backend"))) {
    return path.join(process.resourcesPath, "backend");
  }
  return path.join(__dirname, "..");
}

function ensureDir(targetPath) {
  fs.mkdirSync(targetPath, { recursive: true });
}

function packagesForProfile(backendProfile) {
  const profilePackages = PROFILE_PACKAGES[backendProfile] || PROFILE_PACKAGES.default;
  return [...COMMON_PACKAGES, ...profilePackages];
}

function resetManagedRuntime(paths) {
  fs.rmSync(paths.venvDir, { recursive: true, force: true });
  fs.rmSync(paths.packageRuntimeDir, { recursive: true, force: true });
  fs.rmSync(paths.bootstrapDir, { recursive: true, force: true });
}

function hasBundledWheelhouse(paths) {
  if (!fs.existsSync(paths.bundledWheelhouseDir)) {
    return false;
  }
  try {
    return fs.readdirSync(paths.bundledWheelhouseDir).some((name) => name.endsWith(".whl"));
  } catch (error) {
    return false;
  }
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

function pipInstallArgs(pythonBinary, paths, packages, extraArgs = []) {
  const args = ["pip", "install", "--python", pythonBinary];
  if (hasBundledWheelhouse(paths)) {
    args.push("--find-links", paths.bundledWheelhouseDir);
  }
  return [...args, ...extraArgs, ...packages];
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
    VOICE_FACTORY_WORKSPACE_ROOT: process.env.VOICE_FACTORY_WORKSPACE_ROOT || paths.workspaceDir,
    VOICE_FACTORY_BACKEND_PROFILE: backendProfile,
    VOICE_FACTORY_DEFAULT_COMPUTE_TARGET: defaultComputeTarget,
    VOICE_FACTORY_CUDA_CHANNEL: cudaChannel,
    VOICE_FACTORY_PIPER_RUNTIME_PYTHON: paths.packageRuntimePython.piper,
    VOICE_FACTORY_SBV2_RUNTIME_PYTHON: paths.packageRuntimePython.sbv2,
    VOICE_FACTORY_MIOTTS_RUNTIME_PYTHON: paths.packageRuntimePython.miotts,
    HF_HOME: process.env.HF_HOME || path.join(paths.cacheDir, "huggingface"),
    TORCH_HOME: process.env.TORCH_HOME || path.join(paths.cacheDir, "torch"),
    XDG_CACHE_HOME: process.env.XDG_CACHE_HOME || paths.cacheDir,
    WANDB_MODE: "disabled",
    WANDB_DISABLED: "true",
    WANDB_SILENT: "true",
  };
  return env;
}

function installRuntimePackages(uvCommand, pythonBinary, packages, backendRoot, uvEnv) {
  runChecked(
    uvCommand,
    pipInstallArgs(pythonBinary, { bundledWheelhouseDir: path.join(backendRootForResources(), "wheelhouse") }, packages),
    { env: uvEnv, cwd: backendRoot }
  );
}

function installBundledBootstrapWheels(uvCommand, pythonBinary, paths, backendRoot, uvEnv) {
  if (!hasBundledWheelhouse(paths)) {
    return;
  }
  runChecked(
    uvCommand,
    pipInstallArgs(
      pythonBinary,
      paths,
      WINDOWS_BUNDLED_WHEEL_REQUIREMENTS,
      ["--no-index", "--force-reinstall"]
    ),
    { env: uvEnv, cwd: backendRoot }
  );
}

function prefetchSbv2JapaneseBert(pythonBinary, backendRoot, env) {
  const script = `
from pathlib import Path
from huggingface_hub import snapshot_download
from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages

target = DEFAULT_BERT_MODEL_PATHS[Languages.JP]
target.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(
    "ku-nlp/deberta-v2-large-japanese-char-wwm",
    local_dir=str(target),
)
print(target)
`.trim();
  runChecked(pythonBinary, ["-c", script], { env, cwd: backendRoot });
}

function installPackageRuntimeEnvironments(uvCommand, paths, backendRoot, backendProfile) {
  const uvEnv = {
    ...process.env,
    UV_CACHE_DIR: paths.uvCacheDir,
    UV_PYTHON_INSTALL_DIR: paths.pythonInstallDir,
  };

  for (const runtimeName of Object.keys(PACKAGE_RUNTIME_PACKAGES)) {
    const venvDir = paths.packageRuntimeVenvDir[runtimeName];
    const pythonBinary = paths.packageRuntimePython[runtimeName];
    runChecked(uvCommand, ["venv", venvDir, "--python", MANAGED_PYTHON_VERSION], { env: uvEnv });

    if (runtimeName === "sbv2") {
      if (backendProfile === "nvidia") {
        runChecked(
          uvCommand,
          [
            "pip",
            "install",
            "--python",
            pythonBinary,
            "--index-url",
            NVIDIA_TORCH_INDEX_URL,
            "torch>=2.4.0",
            "torchaudio>=2.4.0",
          ],
          { env: uvEnv, cwd: backendRoot }
        );
      } else {
      runChecked(uvCommand, pipInstallArgs(pythonBinary, paths, ["torch>=2.4.0"]), {
        env: uvEnv,
        cwd: backendRoot,
      });
      }
    }

    installRuntimePackages(uvCommand, pythonBinary, PACKAGE_RUNTIME_PACKAGES[runtimeName], backendRoot, uvEnv);

    if (runtimeName === "sbv2") {
      prefetchSbv2JapaneseBert(pythonBinary, backendRoot, uvEnv);
    }
  }
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
  runChecked(paths.pythonBinary, ["-m", "ensurepip", "--upgrade"], { env: uvEnv, cwd: backendRoot });
  installBundledBootstrapWheels(uvCommand, paths.pythonBinary, paths, backendRoot, uvEnv);

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
    pipInstallArgs(paths.pythonBinary, paths, packagesForProfile(backendProfile)),
    { env: uvEnv, cwd: backendRoot }
  );
  runChecked(
    uvCommand,
    ["pip", "install", "--python", paths.pythonBinary, "--no-deps", "--force-reinstall", "--editable", backendRoot],
    { env: uvEnv, cwd: backendRoot }
  );
  if (backendProfile !== "amd") {
    installPackageRuntimeEnvironments(uvCommand, paths, backendRoot, backendProfile);
  }
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
    resetManagedRuntime(paths);
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
