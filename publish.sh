#!/usr/bin/env bash
# One-shot: push everything to GitHub, set the description, and open-source it
# (make the repo public). Run this on your Mac (where `git` and `gh` are set up):
#
#     bash publish.sh
#
set -euo pipefail
cd "$(dirname "$0")"

REPO="rohanarun/virtual-human"
DESC="Solving problems in the human body by formalizing the real world into code for agents."

echo "==> Pushing commits to GitHub..."
git push origin HEAD

if command -v gh >/dev/null 2>&1; then
  echo "==> Setting repository description..."
  gh repo edit "$REPO" --description "$DESC" || true

  echo "==> Making the repository public (MIT open source)..."
  gh repo edit "$REPO" --visibility public --accept-visibility-change-consequences 2>/dev/null \
    || gh repo edit "$REPO" --visibility public 2>/dev/null \
    || gh api -X PATCH "repos/$REPO" -f private=false >/dev/null

  echo
  echo "Done. Public at: https://github.com/$REPO"
else
  echo
  echo "Pushed. 'gh' (GitHub CLI) not found, so finish these two in the browser or install gh:"
  echo "  - Settings > General > Danger Zone > Change visibility > Public"
  echo "  - About (gear icon) > Description:  $DESC"
fi
