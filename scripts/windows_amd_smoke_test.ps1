param(
  [string]$RepoRoot = "$env:USERPROFILE\Desktop\kvs-src",
  [string]$StateRoot = "C:\kvs-amd-test",
  [string]$RuntimeRoot = "",
  [string]$StyleInstruction = "Calm female narration voice. Natural, clear, and suitable for reading aloud.",
  [ValidateSet("kizuna", "qwen")]
  [string]$SeedVoiceBackend = "kizuna",
  [int]$Port = 7861,
  [int]$PreviewTimeoutSeconds = 900,
  [switch]$RunBuildTts,
  [int]$BuildTimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"

function Wait-Health {
  param(
    [int]$TimeoutSeconds = 60
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      return Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/health"
    } catch {
      Start-Sleep -Seconds 2
    }
  }

  throw "backend did not become ready on port $Port"
}

Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | ForEach-Object {
  Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}

$managedRoot = Join-Path $StateRoot "amd"
if (-not $RuntimeRoot) {
  $RuntimeRoot = $managedRoot
}
$python = Join-Path $RuntimeRoot "runtime\venv\Scripts\python.exe"
$workspaceRoot = Join-Path $managedRoot "workspace"
$cacheRoot = Join-Path $managedRoot "cache"
$logDir = Join-Path $env:USERPROFILE "Desktop\kvs-logs-amd"

New-Item -ItemType Directory -Force $logDir | Out-Null

$env:PYTHONPATH = (Join-Path $RepoRoot "src")
$env:VOICE_FACTORY_WORKSPACE_ROOT = $workspaceRoot
$env:VOICE_FACTORY_MANAGED_STATE_ROOT = $StateRoot
$env:VOICE_FACTORY_BACKEND_PROFILE = "amd"
$env:VOICE_FACTORY_DEFAULT_COMPUTE_TARGET = "cpu"
$env:VOICE_FACTORY_CUDA_CHANNEL = "none"
$env:VOICE_FACTORY_PIPER_RUNTIME_PYTHON = (Join-Path $RuntimeRoot "package-runtimes\piper\venv\Scripts\python.exe")
$env:VOICE_FACTORY_SBV2_RUNTIME_PYTHON = (Join-Path $RuntimeRoot "package-runtimes\sbv2\venv\Scripts\python.exe")
$env:VOICE_FACTORY_MIOTTS_RUNTIME_PYTHON = (Join-Path $RuntimeRoot "package-runtimes\miotts\venv\Scripts\python.exe")
$env:HF_HOME = (Join-Path $cacheRoot "huggingface")
$env:TORCH_HOME = (Join-Path $cacheRoot "torch")
$env:XDG_CACHE_HOME = $cacheRoot
$env:WANDB_MODE = "disabled"
$env:WANDB_DISABLED = "true"
$env:WANDB_SILENT = "true"
$env:VOICE_FACTORY_PIPER_MAX_EPOCHS = "1"
$env:VOICE_FACTORY_PIPER_CHECKPOINT_EPOCHS = "1"
$env:VOICE_FACTORY_PIPER_DISABLE_WAVLM = "1"

$stdoutLog = Join-Path $logDir "stdout.log"
$stderrLog = Join-Path $logDir "stderr.log"

Start-Process `
  -FilePath $python `
  -ArgumentList "-m", "uvicorn", "voice_factory.server:app", "--host", "127.0.0.1", "--port", "$Port" `
  -WorkingDirectory $RepoRoot `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog | Out-Null

$health = Wait-Health -TimeoutSeconds 90
Write-Host "HEALTH"
$health | ConvertTo-Json -Depth 10

$requestBody = @{
  style_instruction = $StyleInstruction
  gpu_memory_gb = 16
  model_family = "piper"
  seed_voice_backend = $SeedVoiceBackend
  compute_target = "cpu"
} | ConvertTo-Json

$startResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:$Port/v1/quick-start" `
  -ContentType "application/json" `
  -Body $requestBody

Write-Host "START_RESPONSE"
$startResponse | ConvertTo-Json -Depth 10

$jobId = $startResponse.job.job_id
$projectId = $startResponse.project.project_id

if (-not $jobId) {
  throw "quick-start did not return job_id"
}

$jobDeadline = (Get-Date).AddSeconds($PreviewTimeoutSeconds)
$job = $null
while ((Get-Date) -lt $jobDeadline) {
  $job = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/v1/jobs/$jobId"
  $jobStatus = if ($null -ne $job.status) { $job.status } else { "unknown" }
  $jobDetail = if ($null -ne $job.detail) { $job.detail } else { "" }
  Write-Host ("JOB_STATUS " + $jobStatus + " / " + $jobDetail)
  if ($job.status -eq "completed" -or $job.status -eq "failed") {
    break
  }
  Start-Sleep -Seconds 5
}

if (-not $job) {
  throw "job polling returned no data"
}

Write-Host "FINAL_JOB"
$job | ConvertTo-Json -Depth 10

if ($job.status -ne "completed") {
  Write-Host "STDERR_TAIL"
  if (Test-Path $stderrLog) {
    Get-Content $stderrLog -Tail 200
  }
  throw "preview job did not complete successfully"
}

$preview = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/v1/projects/$projectId/preview"
Write-Host "PREVIEW"
$preview | ConvertTo-Json -Depth 10

if (-not $RunBuildTts) {
  return
}

$patchScript = Join-Path $env:TEMP "voice_factory_patch_project.py"
@"
import json
from pathlib import Path

path = Path(r"$workspaceRoot") / "projects" / r"$projectId" / "project.json"
data = json.loads(path.read_text(encoding="utf-8"))
data["items_per_category"] = 1
data["gpu_memory_gb"] = 4
data["target_model_family"] = "piper"
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(path)
"@ | Set-Content -Encoding utf8 $patchScript

& $python $patchScript

$buildBody = @{
  mio_base_url = "https://miotts-hybrid-gngpt3r4wq-as.a.run.app"
  model_family = "piper"
  compute_target = "cpu"
} | ConvertTo-Json

$buildResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:$Port/v1/projects/$projectId/build-tts" `
  -ContentType "application/json" `
  -Body $buildBody

Write-Host "BUILD_RESPONSE"
$buildResponse | ConvertTo-Json -Depth 10

$buildJobId = $buildResponse.job_id
if (-not $buildJobId) {
  throw "build-tts did not return job_id"
}

$buildDeadline = (Get-Date).AddSeconds($BuildTimeoutSeconds)
$buildJob = $null
$trainingSeen = $false
while ((Get-Date) -lt $buildDeadline) {
  $buildJob = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/v1/jobs/$buildJobId"
  $stage = if ($null -ne $buildJob.stage) { $buildJob.stage } else { "unknown" }
  $label = if ($null -ne $buildJob.stage_label) { $buildJob.stage_label } else { "" }
  Write-Host ("BUILD_STATUS " + $buildJob.status + " / " + $stage + " / " + $label)
  if ($stage -eq "training") {
    $trainingSeen = $true
  }
  if ($buildJob.status -eq "completed" -or $buildJob.status -eq "failed") {
    break
  }
  if ($trainingSeen) {
    break
  }
  Start-Sleep -Seconds 5
}

if (-not $buildJob) {
  throw "build job polling returned no data"
}

Write-Host "FINAL_BUILD_JOB"
$buildJob | ConvertTo-Json -Depth 10

if (-not $trainingSeen -and $buildJob.status -ne "completed") {
  throw "build-tts did not reach training stage"
}
