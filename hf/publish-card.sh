#!/usr/bin/env bash
# Publish README.md as the Hugging Face model card, with the card metadata on top (2026-11-21).
set -euo pipefail
cd "$(dirname "$0")/.."
card="$(mktemp)"
trap 'rm -f "$card"' EXIT
cat hf/card-metadata.yml README.md > "$card"
hf upload Cyronius/memdraft "$card" README.md --commit-message "${1:-Sync the model card with the GitHub README}"
