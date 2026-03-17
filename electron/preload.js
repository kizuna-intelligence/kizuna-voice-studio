const { contextBridge, ipcRenderer } = require("electron");

const defaultBase = process.env.VOICE_FACTORY_API_BASE || "http://127.0.0.1:7861";

contextBridge.exposeInMainWorld("voiceFactory", {
  defaultApiBase: defaultBase,
  ensureBackend: () => ipcRenderer.invoke("voice-factory:ensure-backend"),
  buildInfo: () => ipcRenderer.invoke("voice-factory:build-info"),
});
