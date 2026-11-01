#!/usr/bin/env bash
# Commit everything in this folder and push to GitHub.
# Usage: ./push.sh "optional commit message"
set -euo pipefail
cd "$(dirname "$0")"

MSG="${1:-Update virtual-human}"
git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -q -F - <<EOF
$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GicaNCQJqfjcaqjLvKgiTg
EOF
  echo "Committed: $MSG"
fi
git push -q origin HEAD
echo "Pushed to $(git remote get-url origin)"
