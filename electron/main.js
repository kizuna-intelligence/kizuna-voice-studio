const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const { ensureManagedBackend } = require("./backend-bootstrap");
const packageJson = require("./package.json");

const defaultBase = process.env.VOICE_FACTORY_API_BASE || "http://127.0.0.1:7861";
const appRoot = path.resolve(__dirname, "..");
const backendRoot = app.isPackaged ? path.join(process.resourcesPath, "backend") : appRoot;
const backendSourceRoot = path.join(backendRoot, "src");
const buildFlavor =
  process.env.VOICE_FACTORY_DISTRIBUTION_FLAVOR || packageJson.voiceFactoryDistributionFlavor || "dev-local";
const backendProfile =
  process.env.VOICE_FACTORY_BACKEND_PROFILE || packageJson.voiceFactoryBackendProfile || "dev";
const defaultComputeTarget =
  process.env.VOICE_FACTORY_DEFAULT_COMPUTE_TARGET || packageJson.voiceFactoryDefaultComputeTarget || "auto";
const cudaChannel = process.env.VOICE_FACTORY_CUDA_CHANNEL || packageJson.voiceFactoryCudaChannel || "none";
const backendLogDir = path.join(app.getPath("temp"), "voice-factory");
const backendStdoutPath = path.join(backendLogDir, "backend.stdout.log");
const backendStderrPath = path.join(backendLogDir, "backend.stderr.log");

let backendProcess = null;
let backendStartPromise = null;

function defaultPythonCandidates() {
  const bundledRoot = path.join(process.resourcesPath || appRoot, "python");
  const activeVenv = process.env.VIRTUAL_ENV || "";
  if (process.platform === "win32") {
    return [
      path.join(bundledRoot, "python.exe"),
      path.join(appRoot, "venv", "Scripts", "python.exe"),
      activeVenv ? path.join(activeVenv, "Scripts", "python.exe") : "",
      "python.exe",
    ];
  }
  return [
    path.join(bundledRoot, "bin", "python3"),
    path.join(bundledRoot, "bin", "python"),
    path.join(appRoot, "venv", "bin", "python"),
    activeVenv ? path.join(activeVenv, "bin", "python") : "",
    activeVenv ? path.join(activeVenv, "bin", "python3") : "",
    "python3",
    "python",
  ];
}

function resolveBackendPython() {
  if (process.env.VOICE_FACTORY_SERVER_PYTHON) {
    return process.env.VOICE_FACTORY_SERVER_PYTHON;
  }
  for (const candidate of defaultPythonCandidates()) {
    if (!candidate) {
      continue;
    }
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return defaultPythonCandidates().at(-1);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function checkBackendHealth() {
  return new Promise((resolve) => {
    const request = http.get(`${defaultBase}/health`, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });
    request.on("error", () => resolve(false));
    request.setTimeout(1000, () => {
      request.destroy();
      resolve(false);
    });
  });
}

async function ensureBackend() {
  if (await checkBackendHealth()) {
    return { apiBase: defaultBase, started: false };
  }
  if (backendStartPromise) {
    await backendStartPromise;
    return { apiBase: defaultBase, started: true };
  }

  backendStartPromise = (async () => {
    fs.mkdirSync(backendLogDir, { recursive: true });
    const stdoutFd = fs.openSync(backendStdoutPath, "a");
    const stderrFd = fs.openSync(backendStderrPath, "a");
    const managedBackend =
      app.isPackaged || process.env.VOICE_FACTORY_FORCE_MANAGED_BACKEND === "1"
        ? await ensureManagedBackend({
            app,
            backendRoot,
            backendProfile,
            buildFlavor,
            defaultComputeTarget,
            cudaChannel,
            appVersion: packageJson.version,
          })
        : null;
    const backendPython = managedBackend?.pythonPath || resolveBackendPython();
    const env = {
      ...process.env,
      ...(managedBackend?.env || {}),
      PYTHONPATH:
        managedBackend?.env?.PYTHONPATH ||
        backendSourceRoot,
      VOICE_FACTORY_BACKEND_PROFILE: backendProfile,
      VOICE_FACTORY_DEFAULT_COMPUTE_TARGET: defaultComputeTarget,
      VOICE_FACTORY_CUDA_CHANNEL: cudaChannel,
    };
    backendProcess = spawn(
      backendPython,
      ["-m", "uvicorn", "voice_factory.server:app", "--host", "127.0.0.1", "--port", "7861"],
      {
        cwd: backendRoot,
        env,
        stdio: ["ignore", stdoutFd, stderrFd],
      }
    );
    backendProcess.once("exit", () => {
      backendProcess = null;
    });

    for (let attempt = 0; attempt < 20; attempt += 1) {
      if (await checkBackendHealth()) {
        return;
      }
      if (backendProcess && backendProcess.exitCode !== null) {
        throw new Error(`Backend exited early. See ${backendStderrPath}`);
      }
      await wait(500);
    }
    throw new Error(`Backend did not become ready. See ${backendStderrPath}`);
  })();

  try {
    await backendStartPromise;
    return { apiBase: defaultBase, started: true };
  } finally {
    backendStartPromise = null;
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 980,
    minWidth: 1360,
    minHeight: 880,
    useContentSize: true,
    backgroundColor: "#ece7de",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile(path.join(__dirname, "index.html"));
}

ipcMain.handle("voice-factory:ensure-backend", async () => ensureBackend());
ipcMain.handle("voice-factory:build-info", async () => ({
  apiBase: defaultBase,
  backendProfile,
  buildFlavor,
  defaultComputeTarget,
  cudaChannel,
  isPackaged: app.isPackaged,
  managedBackend: app.isPackaged || process.env.VOICE_FACTORY_FORCE_MANAGED_BACKEND === "1",
}));

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess && backendProcess.exitCode === null) {
    backendProcess.kill();
  }
});
