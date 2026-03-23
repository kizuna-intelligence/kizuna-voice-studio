#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/diagnose_windows_ssh_auth.sh <target_user> <target_host>
# Example:
#   ./scripts/diagnose_windows_ssh_auth.sh Yusuke 192.168.1.16

TARGET_USER="${1:-Yusuke}"
TARGET_HOST="${2:-192.168.1.16}"

PS_SCRIPT=$(cat <<'POWERSHELL'
$ErrorActionPreference = "Continue"

Write-Host "===== whoami ====="
whoami
Write-Host ""

Write-Host "===== groups ====="
whoami /groups
Write-Host ""

Write-Host "===== user authorized_keys ====="
Get-Content "$env:USERPROFILE\.ssh\authorized_keys" -ErrorAction SilentlyContinue
Write-Host ""

Write-Host "===== admin authorized_keys ====="
Get-Content "C:\ProgramData\ssh\administrators_authorized_keys" -ErrorAction SilentlyContinue
Write-Host ""

Write-Host "===== acl: user .ssh ====="
icacls "$env:USERPROFILE\.ssh"
Write-Host ""

Write-Host "===== acl: user authorized_keys ====="
icacls "$env:USERPROFILE\.ssh\authorized_keys"
Write-Host ""

Write-Host "===== acl: admin authorized_keys ====="
icacls "C:\ProgramData\ssh\administrators_authorized_keys"
Write-Host ""

Write-Host "===== sshd_config ====="
Get-Content "C:\ProgramData\ssh\sshd_config" | Select-String "PubkeyAuthentication|AuthorizedKeysFile|Match Group"
Write-Host ""

Write-Host "===== sshd service ====="
Get-Service sshd
Write-Host ""

Write-Host "===== recent OpenSSH log ====="
Get-WinEvent -LogName "OpenSSH/Operational" -MaxEvents 30 |
  Select-Object TimeCreated, Id, LevelDisplayName, Message |
  Format-List
POWERSHELL
)

ENCODED=$(printf '%s' "$PS_SCRIPT" | iconv -f UTF-8 -t UTF-16LE | base64 -w0)

ssh "${TARGET_USER}@${TARGET_HOST}" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $ENCODED"
