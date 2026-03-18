#!/usr/bin/env bash
set -euo pipefail

# Install the Windows-side software needed to debug and test Kizuna Voice Studio.
# This script runs from Linux/macOS and installs packages on the target Windows
# machine over SSH by invoking winget via PowerShell.

TARGET_USER="${USER:-windows-user}"
TARGET_HOST="windows-host"
INCLUDE_PYTHON=0
AMD_PROFILE=0

usage() {
  cat <<'EOF'
Usage:
  install_windows_test_prereqs.sh [--with-python] [--amd] [user] [host]

Examples:
  install_windows_test_prereqs.sh your-user 192.168.1.10
  install_windows_test_prereqs.sh --with-python your-user 192.168.1.10
  install_windows_test_prereqs.sh --amd your-user 192.168.1.10

Default packages:
  - Git.Git
  - OpenJS.NodeJS.LTS
  - Google.Chrome
  - Microsoft.VCRedist.2015+.x64
  - 7zip.7zip

Optional:
  --with-python   Also install Python 3.11 on the Windows host.
  --amd           Prepare a Windows AMD test machine. This also installs Python 3.11.
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-python)
      INCLUDE_PYTHON=1
      shift
      ;;
    --amd)
      AMD_PROFILE=1
      INCLUDE_PYTHON=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  TARGET_USER="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  TARGET_HOST="${POSITIONAL[1]}"
fi

PACKAGES=(
  "Git.Git"
  "OpenJS.NodeJS.LTS"
  "Google.Chrome"
  "Microsoft.VCRedist.2015+.x64"
  "7zip.7zip"
)

if [[ "$INCLUDE_PYTHON" -eq 1 ]]; then
  PACKAGES+=("Python.Python.3.11")
fi

PACKAGE_LIST=$(printf "'%s'," "${PACKAGES[@]}")
PACKAGE_LIST="@(${PACKAGE_LIST%,})"

read -r -d '' POWERSHELL_SCRIPT <<EOF || true
\$ErrorActionPreference = 'Stop'

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
  throw 'winget is required on the Windows host.'
}

\$packages = $PACKAGE_LIST

foreach (\$pkg in \$packages) {
  Write-Host ('[install] ' + \$pkg)
  winget install -e --id \$pkg --accept-package-agreements --accept-source-agreements --silent
}

Write-Host '--- versions ---'
if (Get-Command git -ErrorAction SilentlyContinue) {
  git --version
}
if (Get-Command node -ErrorAction SilentlyContinue) {
  node --version
}
if (Get-Command python -ErrorAction SilentlyContinue) {
  python --version
}
if (Test-Path 'C:\Program Files\Google\Chrome\Application\chrome.exe') {
  Write-Host 'chrome=installed'
}
if (Test-Path 'C:\Program Files\7-Zip\7z.exe') {
  Write-Host '7zip=installed'
}
EOF

ENCODED_COMMAND=$(
  printf '%s' "$POWERSHELL_SCRIPT" | iconv -f UTF-8 -t UTF-16LE | base64 -w 0
)

echo "installing Windows test prerequisites on ${TARGET_USER}@${TARGET_HOST}"
echo "packages: ${PACKAGES[*]}"
if [[ "$AMD_PROFILE" -eq 1 ]]; then
  echo "profile: amd-compat"
  echo "note: current Windows AMD build uses the compatibility path rather than a dedicated GPU runtime."
fi
echo "password may be requested once"

ssh "${TARGET_USER}@${TARGET_HOST}" \
  "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand ${ENCODED_COMMAND}"

echo "done"
