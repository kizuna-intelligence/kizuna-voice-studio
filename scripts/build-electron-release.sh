#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ELECTRON_DIR="$ROOT_DIR/electron"

usage() {
  cat <<'EOF'
Usage:
  scripts/build-electron-release.sh --variant <variant>

Examples:
  scripts/build-electron-release.sh --variant linux-nvidia
  scripts/build-electron-release.sh --variant windows-nvidia

Notes:
  - Run this on the target OS for the variant you want to publish.
  - This script builds locally and leaves artifacts in electron/dist/<variant>/.
EOF
}

VARIANT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$VARIANT" ]]; then
  echo "--variant is required" >&2
  usage >&2
  exit 1
fi

cd "$ELECTRON_DIR"
npm ci
npm run "dist:$VARIANT"

echo
echo "Build completed."
echo "Artifact directory: $ELECTRON_DIR/dist/$VARIANT"
find "$ELECTRON_DIR/dist/$VARIANT" -maxdepth 1 -type f | sort
