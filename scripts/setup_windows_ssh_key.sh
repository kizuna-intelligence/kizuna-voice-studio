#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_windows_ssh_key.sh [--admin] <target_user> <target_host> [public_key_path]
# Example:
#   ./scripts/setup_windows_ssh_key.sh Yusuke 192.168.1.16
#   ./scripts/setup_windows_ssh_key.sh --admin Yusuke 192.168.1.16
#   ./scripts/setup_windows_ssh_key.sh Yusuke 192.168.1.16 ~/.ssh/id_rsa.pub

ADMIN_MODE=0

if [[ "${1:-}" == "--admin" ]]; then
  ADMIN_MODE=1
  shift
fi

TARGET_USER="${1:-Yusuke}"
TARGET_HOST="${2:-192.168.1.16}"
PUBKEY="${3:-$HOME/.ssh/id_ed25519.pub}"
PRIVATE_KEY="${PUBKEY%.pub}"

if [[ ! -f "$PUBKEY" ]]; then
  echo "public key not found: $PUBKEY" >&2
  exit 1
fi

if [[ ! -f "$PRIVATE_KEY" ]]; then
  echo "private key not found: $PRIVATE_KEY" >&2
  exit 1
fi

PUBKEY_CONTENT=$(tr -d '\r' < "$PUBKEY")
PUBKEY_B64=$(printf '%s' "$PUBKEY_CONTENT" | base64 -w0)

echo "sending key: $PUBKEY -> ${TARGET_USER}@${TARGET_HOST}"
if [[ "$ADMIN_MODE" -eq 1 ]]; then
  echo "target mode: administrators_authorized_keys"
else
  echo "target mode: authorized_keys"
fi
echo "password may be requested once while installing the key"

PS_SCRIPT=$(cat <<POWERSHELL
\$ErrorActionPreference = "Stop"
\$key = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String("$PUBKEY_B64")).Trim()
\$userDir = Join-Path \$env:USERPROFILE ".ssh"
\$userAuth = Join-Path \$userDir "authorized_keys"

New-Item -ItemType Directory -Force \$userDir | Out-Null

function Add-KeyIfMissing {
  param([string]\$Path, [string]\$Value)
  if (Test-Path \$Path) {
    \$existing = Get-Content \$Path -Raw
    if (\$existing -notmatch [regex]::Escape(\$Value)) {
      Add-Content -Encoding ascii -Path \$Path -Value \$Value
    }
  } else {
    Set-Content -Encoding ascii -Path \$Path -Value \$Value
  }
}

Add-KeyIfMissing -Path \$userAuth -Value \$key

icacls \$userDir /inheritance:r | Out-Null
icacls \$userDir /grant "\$env:USERNAME`:(F)" | Out-Null
icacls \$userDir /grant "SYSTEM:F" | Out-Null
icacls \$userDir /grant "Administrators:F" | Out-Null

icacls \$userAuth /inheritance:r | Out-Null
icacls \$userAuth /grant "\$env:USERNAME`:(F)" | Out-Null
icacls \$userAuth /grant "SYSTEM:F" | Out-Null
icacls \$userAuth /grant "Administrators:F" | Out-Null

if ($ADMIN_MODE -eq 1) {
  \$adminAuth = "C:\ProgramData\ssh\administrators_authorized_keys"
  Add-KeyIfMissing -Path \$adminAuth -Value \$key
  icacls \$adminAuth /inheritance:r | Out-Null
  icacls \$adminAuth /grant "Administrators:F" | Out-Null
  icacls \$adminAuth /grant "SYSTEM:F" | Out-Null
  Write-Host "administrators_authorized_keys updated:"
  Write-Host \$adminAuth
}

Restart-Service sshd -ErrorAction SilentlyContinue
Write-Host "authorized_keys updated:"
Write-Host \$userAuth
POWERSHELL
)

ENCODED=$(printf '%s' "$PS_SCRIPT" | iconv -f UTF-8 -t UTF-16LE | base64 -w0)

ssh "${TARGET_USER}@${TARGET_HOST}" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $ENCODED"

echo "done"
echo "testing public-key login..."

if ssh -o BatchMode=yes -o PreferredAuthentications=publickey -i "$PRIVATE_KEY" "${TARGET_USER}@${TARGET_HOST}" exit; then
  echo "public-key login: OK"
else
  echo "public-key login: FAILED" >&2
  echo "run the diagnose script next:" >&2
  echo "/home/yusuke/gitrepos/kizuna-voice-studio/scripts/diagnose_windows_ssh_auth.sh ${TARGET_USER} ${TARGET_HOST}" >&2
  exit 2
fi

echo "manual test:"
echo "ssh -i $PRIVATE_KEY ${TARGET_USER}@${TARGET_HOST}"
