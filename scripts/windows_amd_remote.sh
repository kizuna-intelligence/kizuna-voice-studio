#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_HOST="${REMOTE_HOST:-windows-amd-host}"
REMOTE_ROOT="${REMOTE_ROOT:-C:/Users/${REMOTE_USER}/Desktop/kvs-src}"
REMOTE_ELECTRON_DIR="${REMOTE_ROOT}/electron"
REMOTE_SCRIPT_DIR="${REMOTE_ROOT}/scripts"
REMOTE_BUILD_SCRIPT="${REMOTE_SCRIPT_DIR}/windows_amd_smoke_test.ps1"
REMOTE_ZIP_PATH="${REMOTE_ZIP_PATH:-C:/Users/${REMOTE_USER}/Desktop/kizuna-voice-studio-windows-amd-src.zip}"
usage() {
  cat <<'EOF'
Usage:
  scripts/windows_amd_remote.sh <command>

Commands:
  sync               Zip current repo and upload/extract it on the Windows host
  build              Run npm ci and build the windows-amd Electron app on Windows
  smoke-preview      Run the Windows AMD smoke test through preview completion
  smoke-build-tts    Run the Windows AMD smoke test through preview + build-tts/training-stage
  all                Run sync + build + smoke-build-tts

Environment overrides:
  REMOTE_USER        SSH user on the Windows machine
  REMOTE_HOST        SSH host for the Windows machine
  REMOTE_ROOT        Target source checkout on Windows
  REMOTE_ZIP_PATH    Temporary source zip path on Windows

Examples:
  REMOTE_USER=your-user REMOTE_HOST=192.168.1.10 scripts/windows_amd_remote.sh sync
  REMOTE_USER=your-user REMOTE_HOST=192.168.1.10 scripts/windows_amd_remote.sh build
  REMOTE_USER=your-user REMOTE_HOST=192.168.1.10 scripts/windows_amd_remote.sh smoke-preview
  REMOTE_USER=your-user REMOTE_HOST=192.168.1.10 scripts/windows_amd_remote.sh smoke-build-tts
  REMOTE_USER=your-user REMOTE_HOST=192.168.1.10 scripts/windows_amd_remote.sh all
EOF
}

run_remote_cmd() {
  local cmd="$1"
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "cmd /c \"$cmd\""
}

sync_repo() {
  local zip_path
  zip_path="$(mktemp --suffix=.zip)"
  trap 'rm -f "$zip_path"' RETURN

  cd "$ROOT_DIR"
  python - <<'PY' "$zip_path"
import pathlib
import zipfile
import sys

root = pathlib.Path.cwd()
dest = pathlib.Path(sys.argv[1])
exclude_dirs = {
    ".git",
    "node_modules",
    "dist",
    "workspace",
    "__pycache__",
    ".pytest_cache",
}
exclude_suffixes = {".pyc", ".pyo"}

with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix in exclude_suffixes:
            continue
        zf.write(path, rel.as_posix())
PY

  scp "$zip_path" "${REMOTE_USER}@${REMOTE_HOST}:/${REMOTE_ZIP_PATH}"
  run_remote_cmd "if exist ${REMOTE_ROOT//\//\\} rmdir /s /q ${REMOTE_ROOT//\//\\} && mkdir ${REMOTE_ROOT//\//\\} && powershell -NoProfile -Command \"Expand-Archive -Force '${REMOTE_ZIP_PATH//\//\\}' '${REMOTE_ROOT//\//\\}'\""
  echo "synced repo to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
}

build_windows_amd() {
  run_remote_cmd "cd /d ${REMOTE_ELECTRON_DIR//\//\\} && npm.cmd ci && npm.cmd run dist:windows-amd"
}

smoke_preview() {
  ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "powershell -NoProfile -ExecutionPolicy Bypass -File ${REMOTE_BUILD_SCRIPT//\//\\}"
}

smoke_build_tts() {
  ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "powershell -NoProfile -ExecutionPolicy Bypass -File ${REMOTE_BUILD_SCRIPT//\//\\} -RunBuildTts"
}

command="${1:-}"
case "$command" in
  sync)
    sync_repo
    ;;
  build)
    build_windows_amd
    ;;
  smoke-preview)
    smoke_preview
    ;;
  smoke-build-tts)
    smoke_build_tts
    ;;
  all)
    sync_repo
    build_windows_amd
    smoke_build_tts
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "unknown command: $command" >&2
    usage >&2
    exit 1
    ;;
esac
