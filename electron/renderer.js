const apiBaseInput = document.getElementById("apiBase");
const output = document.getElementById("output");
const projectIdInput = document.getElementById("projectId");
const voiceDescriptionInput = document.getElementById("voiceDescription");
const gpuMemoryInput = document.getElementById("gpuMemoryGb");
const seedVoiceBackendSelect = document.getElementById("seedVoiceBackend");
const seedVoiceBackendHint = document.getElementById("seedVoiceBackendHint");
const computeTargetSelect = document.getElementById("computeTarget");
const computeTargetHint = document.getElementById("computeTargetHint");
const buildVariantNote = document.getElementById("buildVariantNote");
const previewAudio = document.getElementById("previewAudio");
const statusLine = document.getElementById("statusLine");
const progressPanel = document.getElementById("progressPanel");
const progressTitle = document.getElementById("progressTitle");
const progressValue = document.getElementById("progressValue");
const progressFill = document.getElementById("progressFill");
const previewButton = document.getElementById("previewButton");
const buildTtsButton = document.getElementById("buildTtsButton");
const downloadButton = document.getElementById("downloadButton");
const helpButton = document.getElementById("helpButton");
const closeHelpButton = document.getElementById("closeHelpButton");
const helpOverlay = document.getElementById("helpOverlay");
const modelPanel = document.getElementById("modelPanel");
const modelPanelLead = document.getElementById("modelPanelLead");
const modelRecommendation = document.getElementById("modelRecommendation");
const trainingStageDescription = document.getElementById("trainingStageDescription");
const packageStageDescription = document.getElementById("packageStageDescription");
const previewStageDescription = document.getElementById("previewStageDescription");
const modelFamilyInputs = Array.from(document.querySelectorAll('input[name="modelFamily"]'));
const stageCards = Array.from(document.querySelectorAll("[data-stage-card]"));

const defaultMioBaseUrl = "https://miotts-hybrid-gngpt3r4wq-as.a.run.app";
const computeTargetStorageKey = "voiceFactory.computeTarget";
const stageOrder = ["preview", "dataset", "training", "package"];
const maleVoiceHintKeywords = [
  "男性",
  "男声",
  "低い声",
  "低め",
  "渋い",
  "ベテラン",
  "刑事",
  "おじさん",
  "青年男性",
  "中年男性",
];
const modelCopy = {
  piper: {
    label: "Piper TTS",
    training: "Piper を学習して、軽量で配布しやすいモデルに整えます。",
    package: "Piper を `pip install` できる Python パッケージにまとめます。",
    download: "完成した Piper パッケージをダウンロード",
  },
  sbv2: {
    label: "Style-Bert-VITS2",
    training: "Style-Bert-VITS2 を学習して、より表現力の高いモデルに整えます。",
    package: "Style-Bert-VITS2 を `pip install` できる Python パッケージにまとめます。",
    download: "完成した Style-Bert-VITS2 パッケージをダウンロード",
  },
};
const seedVoiceCopy = {
  kizuna: {
    label: "Kizuna Voice Designer",
    hint: "日本語をそのまま使う lightweight モードです。",
    stage: "Kizuna Voice Designer の lightweight モードで基準となる声を作ります。",
  },
  qwen: {
    label: "Qwen Voice Designer",
    hint: "日本語を中国語へ変換してから Qwen VoiceDesign で種音声を作ります。",
    stage: "Qwen VoiceDesign と翻訳モデルで基準となる声を作ります。",
  },
};

let previewReady = false;
let buildInfo = {
  buildFlavor: "dev-local",
  backendProfile: "dev",
  defaultComputeTarget: "auto",
  cudaChannel: "none",
};

apiBaseInput.value = window.voiceFactory?.defaultApiBase || "http://127.0.0.1:7861";
voiceDescriptionInput.value =
  "20代後半の女性。落ち着いていて明るすぎず、ニュースを自然に読める聞き取りやすい声。";

function fallbackComputeTargets(profile = buildInfo.backendProfile) {
  if (profile === "cpu") {
    return [{ value: "cpu", label: "CPU", description: "この配布版は CPU 実行を前提にしています。" }];
  }
  if (profile === "apple") {
    return [
      { value: "auto", label: "自動で選ぶ", description: "Apple Silicon の実行環境を自動で使います。" },
      { value: "cpu", label: "CPU", description: "GPU を使わず CPU で実行します。" },
    ];
  }
  return [
    { value: "auto", label: "自動で選ぶ", description: "利用可能な GPU を自動で使います。" },
    { value: "cpu", label: "CPU", description: "GPU を使わず CPU で実行します。" },
    { value: "gpu:0", label: "GPU 0", description: "GPU 0 を使います。" },
    { value: "gpu:1", label: "GPU 1", description: "GPU 1 を使います。" },
    { value: "gpu:2", label: "GPU 2", description: "GPU 2 を使います。" },
    { value: "gpu:3", label: "GPU 3", description: "GPU 3 を使います。" },
  ];
}

function apiUrl(path) {
  return `${apiBaseInput.value.replace(/\/$/, "")}${path}`;
}

function selectedModelFamily() {
  return modelFamilyInputs.find((input) => input.checked)?.value || "piper";
}

function selectedComputeTarget() {
  return computeTargetSelect.value || "auto";
}

function selectedSeedVoiceBackend() {
  return seedVoiceBackendSelect.value || "kizuna";
}

function buildVariantSummary(info) {
  if (!info || !buildVariantNote) {
    return "";
  }
  if (info.backendProfile === "nvidia") {
    return `この配布版は NVIDIA GPU 向けです。CUDA は新しい系統のみを対象にしています。初回起動時は必要な Python 環境を自動で準備し、既定の実行先は ${info.defaultComputeTarget.toUpperCase()} です。`;
  }
  if (info.backendProfile === "cpu") {
    return "この配布版は CPU 専用です。初回起動時に必要な Python 環境を自動で準備し、GPU 学習や GPU 推論は含めません。";
  }
  if (info.backendProfile === "apple") {
    return "この配布版は Apple Silicon 向けです。初回起動時に必要な Python 環境を自動で準備し、ローカルの Apple Silicon 実行環境に合わせて起動します。";
  }
  return "この配布版は開発用です。実行環境に応じてバックエンドを起動します。";
}

function updateBuildVariantNote() {
  const message = buildVariantSummary(buildInfo);
  if (!message || !buildVariantNote) {
    return;
  }
  buildVariantNote.textContent = message;
  buildVariantNote.classList.remove("hidden");
}

function updateComputeTargetHint() {
  const selectedOption = computeTargetSelect.selectedOptions[0];
  computeTargetHint.textContent =
    selectedOption?.dataset.description || "起動時に CPU か使う GPU を選べます。";
  window.localStorage?.setItem(computeTargetStorageKey, selectedComputeTarget());
}

function renderComputeTargets(targets) {
  const incomingTargets = targets?.length ? targets : fallbackComputeTargets();
  const availableTargets =
    buildInfo.backendProfile === "cpu"
      ? incomingTargets.filter((target) => target.value === "cpu")
      : buildInfo.backendProfile === "apple"
        ? incomingTargets.filter((target) => target.value === "auto" || target.value === "cpu")
        : incomingTargets;
  const normalizedTargets = availableTargets.length ? availableTargets : fallbackComputeTargets();
  const savedTarget = window.localStorage?.getItem(computeTargetStorageKey);
  const preferredTarget =
    normalizedTargets.find((target) => target.value === savedTarget)?.value ||
    normalizedTargets.find((target) => target.value === buildInfo.defaultComputeTarget)?.value ||
    normalizedTargets.find((target) => target.value === "gpu:0")?.value ||
    normalizedTargets[0].value;

  computeTargetSelect.innerHTML = "";
  normalizedTargets.forEach((target) => {
    const option = document.createElement("option");
    option.value = target.value;
    option.textContent = target.label;
    option.dataset.description = target.description || "";
    computeTargetSelect.appendChild(option);
  });
  computeTargetSelect.value = preferredTarget;
  updateComputeTargetHint();
}

async function loadComputeTargets() {
  try {
    const response = await fetch(apiUrl("/v1/system/compute-targets"));
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const payload = await response.json();
    renderComputeTargets(payload.targets);
  } catch (error) {
    renderComputeTargets(fallbackComputeTargets());
  }
}

async function loadBuildInfo() {
  if (typeof window.voiceFactory?.buildInfo !== "function") {
    updateBuildVariantNote();
    return;
  }
  try {
    buildInfo = {
      ...buildInfo,
      ...(await window.voiceFactory.buildInfo()),
    };
    apiBaseInput.value = buildInfo.apiBase || apiBaseInput.value;
  } catch (error) {
    // Keep the local defaults when build metadata is unavailable.
  }
  updateBuildVariantNote();
}

function updateModelCopy() {
  const copy = modelCopy[selectedModelFamily()];
  trainingStageDescription.textContent = copy.training;
  packageStageDescription.textContent = copy.package;
  downloadButton.textContent = copy.download;
  updateModelRecommendation();
}

function updateSeedVoiceBackendCopy() {
  const copy = seedVoiceCopy[selectedSeedVoiceBackend()] || seedVoiceCopy.kizuna;
  seedVoiceBackendHint.textContent = copy.hint;
  previewStageDescription.textContent = copy.stage;
}

function looksLikeMaleVoice(description) {
  const normalized = description.trim();
  return maleVoiceHintKeywords.some((keyword) => normalized.includes(keyword));
}

function updateModelRecommendation() {
  if (!looksLikeMaleVoice(voiceDescriptionInput.value || "")) {
    modelRecommendation.textContent = "";
    modelRecommendation.classList.add("hidden");
    modelRecommendation.classList.remove("warning-note", "info-note");
    return;
  }

  const prefersSbv2 = selectedModelFamily() === "sbv2";
  modelRecommendation.textContent = prefersSbv2
    ? "低めの男性声は、今の比較では Style-Bert-VITS2 の方が安定しやすいです。"
    : "低めの男性声は、今の比較では Style-Bert-VITS2 の方が安定しやすいです。迷ったら Style-Bert-VITS2 を選んでください。";
  modelRecommendation.classList.remove("hidden", "warning-note", "info-note");
  modelRecommendation.classList.add(prefersSbv2 ? "info-note" : "warning-note");
}

function setModelSelectionEnabled(isEnabled) {
  modelPanel.classList.toggle("locked-panel", !isEnabled);
  modelPanelLead.textContent = isEnabled
    ? "声サンプルを確認できたので、ここで作るモデルを選んで TTS 作成へ進めます。"
    : "まず上で種音声を作って確認してください。モデル選択はそのあとです。";
  modelFamilyInputs.forEach((input) => {
    input.disabled = !isEnabled;
    input.closest(".model-option")?.classList.toggle("disabled", !isEnabled);
  });
}

async function request(path, options = {}) {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await fetch(apiUrl(path), {
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        ...options,
      });
      const text = await response.text();
      if (!response.ok) {
        throw new Error(text || `${response.status} ${response.statusText}`);
      }
      return text ? JSON.parse(text) : {};
    } catch (error) {
      const isLastAttempt = attempt === 1;
      if (isLastAttempt || typeof window.voiceFactory?.ensureBackend !== "function") {
        throw error;
      }
      setStatus("backend を準備しています。初回起動時は Python と依存パッケージのセットアップが走ります。");
      await window.voiceFactory.ensureBackend();
    }
  }
}

function setOutput(payload) {
  output.textContent = JSON.stringify(payload, null, 2);
}

function setStatus(message) {
  statusLine.textContent = message;
}

function hideProgress() {
  progressPanel.classList.add("hidden");
  progressTitle.textContent = "進行状況";
  progressValue.textContent = "0%";
  progressFill.style.width = "0%";
}

function showProgress(title, percent) {
  const normalized = Math.max(0, Math.min(100, Math.round(percent)));
  progressPanel.classList.remove("hidden");
  progressTitle.textContent = title;
  progressValue.textContent = `${normalized}%`;
  progressFill.style.width = `${normalized}%`;
}

function previewStagePercent(message) {
  if (!message) {
    return 8;
  }
  if (message.includes("翻訳モデル")) {
    return 20;
  }
  if (message.includes("音声デザイナーモデル")) {
    return 20;
  }
  if (message.includes("VoiceDesign モデル")) {
    return 52;
  }
  if (message.includes("種音声を生成中")) {
    return 82;
  }
  if (message.includes("完了")) {
    return 100;
  }
  return 12;
}

function pipelineStagePercent(payload) {
  if (!payload) {
    return 0;
  }
  if (payload.status === "completed") {
    return 100;
  }
  if (payload.progress_percent !== undefined && payload.progress_percent !== null) {
    const baseByStage = {
      dataset: 20,
      scripts: 48,
    };
    const spanByStage = {
      dataset: 28,
      scripts: 4,
    };
    const base = baseByStage[payload.stage] ?? 0;
    const span = spanByStage[payload.stage] ?? 0;
    return base + Math.round((span * payload.progress_percent) / 100);
  }
  switch (payload.stage) {
    case "preview":
      return 10;
    case "dataset":
      return 24;
    case "scripts":
      return 50;
    case "preprocess":
      return 56;
    case "training":
      return 72;
    case "export":
      return 88;
    case "package":
      return 96;
    default:
      return 8;
  }
}

function formatJobStatus(payload) {
  if (!payload) {
    return "";
  }
  if (payload.progress?.current !== undefined && payload.progress?.total !== undefined) {
    const percent =
      payload.progress_percent !== undefined
        ? payload.progress_percent
        : Math.round((payload.progress.current / Math.max(1, payload.progress.total)) * 100);
    const detail = payload.progress_detail ? ` - ${payload.progress_detail}` : "";
    const base = payload.stage_label_base || payload.stage_label || "進行中です";
    return `${base} (${payload.progress.current}/${payload.progress.total}, ${percent}%)${detail}`;
  }
  return payload.stage_label || "";
}

function setButtonsState({
  previewBusy = false,
  buildBusy = false,
  previewEnabled = true,
  buildEnabled = false,
} = {}) {
  previewButton.disabled = previewBusy || !previewEnabled;
  buildTtsButton.disabled = buildBusy || !buildEnabled;
  previewButton.textContent = previewBusy ? "種音声を生成中..." : "1. 種音声を作成";
  buildTtsButton.textContent = buildBusy ? "TTS を作成中..." : "2. この声で TTS を作る";
}

function setStageState(stateMap = {}) {
  stageCards.forEach((card) => {
    card.classList.remove("active", "done", "idle");
    card.classList.add(stateMap[card.dataset.stageCard] || "idle");
  });
}

function stageGroup(stage) {
  if (stage === "preview") {
    return "preview";
  }
  if (stage === "dataset" || stage === "scripts") {
    return "dataset";
  }
  if (stage === "preprocess" || stage === "training" || stage === "export") {
    return "training";
  }
  if (stage === "package") {
    return "package";
  }
  return null;
}

function setInitialStageBoard() {
  setStageState({
    preview: "idle",
    dataset: "idle",
    training: "idle",
    package: "idle",
  });
}

function setPreviewReadyStageBoard() {
  setStageState({
    preview: "done",
    dataset: "idle",
    training: "idle",
    package: "idle",
  });
}

function updatePipelineStageBoard(stage, status) {
  const currentGroup = stageGroup(stage);
  const currentIndex = currentGroup ? stageOrder.indexOf(currentGroup) : -1;
  const stateMap = {};
  stageOrder.forEach((cardStage, cardIndex) => {
    if (status === "completed") {
      stateMap[cardStage] = "done";
      return;
    }
    if (cardIndex < currentIndex) {
      stateMap[cardStage] = "done";
      return;
    }
    if (cardStage === currentGroup && status === "running") {
      stateMap[cardStage] = "active";
      return;
    }
    stateMap[cardStage] = "idle";
  });
  setStageState(stateMap);
}

function clearPreview() {
  previewReady = false;
  previewAudio.removeAttribute("src");
  previewAudio.load();
}

function setPreviewAudio(projectId) {
  previewAudio.src = `${apiUrl(`/v1/projects/${encodeURIComponent(projectId)}/preview/audio`)}?t=${Date.now()}`;
  previewAudio.load();
}

function setDownloadReady(projectId, manifest, modelFamily) {
  if (!manifest || !manifest.archive_path) {
    downloadButton.classList.add("hidden");
    downloadButton.removeAttribute("href");
    downloadButton.removeAttribute("download");
    return;
  }

  const path =
    modelFamily === "sbv2"
      ? `/v1/projects/${encodeURIComponent(projectId)}/package/sbv2/download`
      : `/v1/projects/${encodeURIComponent(projectId)}/package/download`;
  downloadButton.href = apiUrl(path);
  downloadButton.download = manifest.archive_path.split("/").pop();
  downloadButton.classList.remove("hidden");
}

function openHelpOverlay() {
  helpOverlay.classList.remove("hidden");
}

function closeHelpOverlay() {
  helpOverlay.classList.add("hidden");
}

async function hydrateProject(projectId, { quietOutput = false } = {}) {
  const payload = await request(`/v1/projects/${encodeURIComponent(projectId)}`);
  const activeModelFamily = payload.project?.target_model_family || selectedModelFamily();
  if (!quietOutput) {
    setOutput(payload);
  }
  if (payload.preview && payload.preview.reference_wav) {
    previewReady = true;
    setPreviewAudio(projectId);
    if (!payload.package?.ready && !payload.sbv2_package?.ready) {
      setPreviewReadyStageBoard();
      setButtonsState({
        previewEnabled: true,
        buildEnabled: true,
      });
      setModelSelectionEnabled(true);
    }
  }

  const activePackage =
    activeModelFamily === "sbv2" ? payload.sbv2_package?.manifest : payload.package?.manifest;
  setDownloadReady(projectId, activePackage, activeModelFamily);

  if ((activeModelFamily === "piper" && payload.package?.ready) || (activeModelFamily === "sbv2" && payload.sbv2_package?.ready)) {
    previewReady = true;
    setStageState({
      preview: "done",
      dataset: "done",
      training: "done",
      package: "done",
    });
    setButtonsState({
      previewEnabled: true,
      buildEnabled: true,
    });
    setModelSelectionEnabled(true);
  }
  return payload;
}

async function pollJob(jobId, projectId, { mode }) {
  for (;;) {
    const payload = await request(`/v1/jobs/${encodeURIComponent(jobId)}`);
    setOutput(payload);
    const statusMessage = formatJobStatus(payload);
    if (statusMessage) {
      setStatus(statusMessage);
    }
    if (mode === "preview") {
      showProgress("種音声の作成", previewStagePercent(payload.stage_label));
    } else {
      showProgress("TTS の作成", pipelineStagePercent(payload));
    }
    if (projectId && payload.stage !== "queued") {
      try {
        await hydrateProject(projectId, { quietOutput: true });
      } catch (error) {
        // Ignore transient 404s while artifacts are still being written.
      }
    }
    if (payload.status === "completed") {
      await hydrateProject(projectId);
      if (mode === "preview") {
        showProgress("種音声の作成", 100);
        setPreviewReadyStageBoard();
        setButtonsState({
          previewEnabled: true,
          buildEnabled: true,
        });
        setModelSelectionEnabled(true);
        setStatus("声サンプルができました。再生して確認し、よければ「この声で TTS を作る」を押してください。");
      } else {
        showProgress("TTS の作成", 100);
        setStageState({
          preview: "done",
          dataset: "done",
          training: "done",
          package: "done",
        });
        setButtonsState({
          previewEnabled: true,
          buildEnabled: true,
        });
        setModelSelectionEnabled(true);
        setStatus(`${modelCopy[selectedModelFamily()].label} のパッケージを用意しました。下のボタンからダウンロードできます。`);
      }
      return payload;
    }
    if (payload.status === "failed") {
      throw new Error(payload.result?.error || `Job failed: ${jobId}`);
    }
    if (mode === "preview") {
      setStageState({
        preview: "active",
        dataset: "idle",
        training: "idle",
        package: "idle",
      });
    } else {
      updatePipelineStageBoard(payload.stage, payload.status);
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

helpButton.addEventListener("click", openHelpOverlay);
closeHelpButton.addEventListener("click", closeHelpOverlay);
helpOverlay.addEventListener("click", (event) => {
  if (event.target === helpOverlay) {
    closeHelpOverlay();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeHelpOverlay();
  }
});

modelFamilyInputs.forEach((input) => {
  input.addEventListener("change", async () => {
    updateModelCopy();
    if (projectIdInput.value) {
      try {
        await hydrateProject(projectIdInput.value, { quietOutput: true });
      } catch (error) {
        // Ignore refresh failures during active jobs.
      }
    }
  });
});

computeTargetSelect.addEventListener("change", updateComputeTargetHint);
seedVoiceBackendSelect.addEventListener("change", updateSeedVoiceBackendCopy);
voiceDescriptionInput.addEventListener("input", updateModelRecommendation);

previewButton.addEventListener("click", async () => {
  try {
    setButtonsState({
      previewBusy: true,
      previewEnabled: true,
      buildEnabled: false,
    });
    clearPreview();
    projectIdInput.value = "";
    setDownloadReady("", null, selectedModelFamily());
    setInitialStageBoard();
    setModelSelectionEnabled(false);
    setStatus("種音声を生成しています。しばらく待ってください。");
    showProgress("種音声の作成", 8);
    const payload = await request("/v1/quick-start", {
      method: "POST",
      body: JSON.stringify({
        style_instruction: voiceDescriptionInput.value,
        gpu_memory_gb: Number(gpuMemoryInput.value || 16),
        model_family: selectedModelFamily(),
        seed_voice_backend: selectedSeedVoiceBackend(),
        compute_target: selectedComputeTarget(),
      }),
    });
    const projectId = payload.project.project_id;
    projectIdInput.value = projectId;
    setOutput(payload);
    await pollJob(payload.job.job_id, projectId, { mode: "preview" });
  } catch (error) {
    setInitialStageBoard();
    setModelSelectionEnabled(false);
    hideProgress();
    setStatus("種音声の生成に失敗しました。詳細ログを確認してください。");
    setOutput({ error: String(error) });
  } finally {
    setButtonsState({
      previewEnabled: true,
      buildEnabled: previewReady,
    });
  }
});

buildTtsButton.addEventListener("click", async () => {
  const projectId = projectIdInput.value;
  const modelFamily = selectedModelFamily();
  if (!projectId) {
    setStatus("先に種音声を作成して、声サンプルを確認してください。");
    return;
  }

  try {
    setButtonsState({
      previewEnabled: true,
      buildBusy: true,
      buildEnabled: true,
    });
    setDownloadReady("", null, modelFamily);
    setPreviewReadyStageBoard();
    setStatus(`確認済みの種音声を使って ${modelCopy[modelFamily].label} を作成しています。`);
    showProgress("TTS の作成", 12);
    const payload = await request(`/v1/projects/${encodeURIComponent(projectId)}/build-tts`, {
      method: "POST",
      body: JSON.stringify({
        mio_base_url: defaultMioBaseUrl,
        model_family: modelFamily,
        compute_target: selectedComputeTarget(),
      }),
    });
    setOutput(payload);
    await pollJob(payload.job_id, projectId, { mode: "build" });
  } catch (error) {
    hideProgress();
    setStatus("TTS 作成に失敗しました。詳細ログを確認してください。");
    setOutput({ error: String(error) });
  } finally {
    setButtonsState({
      previewEnabled: true,
      buildEnabled: previewReady,
    });
  }
});

updateModelCopy();
updateSeedVoiceBackendCopy();
updateModelRecommendation();
setInitialStageBoard();
setModelSelectionEnabled(false);
hideProgress();
setButtonsState({
  previewEnabled: true,
  buildEnabled: false,
});
loadBuildInfo().finally(() => loadComputeTargets());
