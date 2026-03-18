#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/electron/dist"
DEFAULT_REPO="kizuna-intelligence/kizuna-voice-studio"
GITHUB_CLI="${GITHUB_CLI:-aih-gh}"

usage() {
  cat <<'EOF'
Usage:
  scripts/upload-electron-release-assets.sh --tag <tag> --variant <variant> [--repo owner/name]

Examples:
  scripts/upload-electron-release-assets.sh --tag v0.1.1 --variant linux-nvidia
  scripts/upload-electron-release-assets.sh --tag v0.1.1 --variant windows-nvidia
  scripts/upload-electron-release-assets.sh --tag v0.1.1 --variant windows-amd

Notes:
  - Build locally first with scripts/build-electron-release.sh.
  - If the release does not exist yet, this script creates it.
  - Existing assets with the same name are overwritten.
EOF
}

TAG=""
VARIANT=""
REPO="$DEFAULT_REPO"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
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

if [[ -z "$TAG" || -z "$VARIANT" ]]; then
  echo "--tag and --variant are required" >&2
  usage >&2
  exit 1
fi

ARTIFACT_DIR="$DIST_DIR/$VARIANT"
if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "Artifact directory not found: $ARTIFACT_DIR" >&2
  echo "Run scripts/build-electron-release.sh first." >&2
  exit 1
fi

if ! command -v "$GITHUB_CLI" >/dev/null 2>&1; then
  if command -v gh >/dev/null 2>&1; then
    GITHUB_CLI="gh"
  else
    echo "Neither $GITHUB_CLI nor gh is available on PATH." >&2
    exit 1
  fi
fi

mapfile -t ASSETS < <(
  find "$ARTIFACT_DIR" -maxdepth 1 -type f \
    \( -name '*.AppImage' -o -name '*.exe' -o -name '*.dmg' -o -name '*.zip' -o -name '*.yml' -o -name '*.blockmap' \) \
    | sort
)

if [[ ${#ASSETS[@]} -eq 0 ]]; then
  echo "No release assets found in $ARTIFACT_DIR" >&2
  exit 1
fi

if ! "$GITHUB_CLI" release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  "$GITHUB_CLI" release create "$TAG" \
    --repo "$REPO" \
    --title "$TAG" \
    --notes "Kizuna Voice Studio $TAG"
fi

"$GITHUB_CLI" release upload "$TAG" "${ASSETS[@]}" --repo "$REPO" --clobber

echo
echo "Uploaded assets to $REPO release $TAG"
printf ' - %s\n' "${ASSETS[@]}"
