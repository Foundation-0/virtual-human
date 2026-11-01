#!/usr/bin/env bash
# Creates the private GitHub repo for the Virtual Human project.
# Run on your Mac:  bash create_github_repo.sh
set -euo pipefail

OWNER="rohanarun"
REPO="virtual-human"
DESC="Multi-scale virtual human: from-scratch atomistic engine (JAX) + whole-body physiology, with a real-time browser demo."

# 1. GitHub CLI
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) not found."
  if command -v brew >/dev/null 2>&1; then
    echo "Installing it with Homebrew..."
    brew install gh
  else
    echo "Install it from https://cli.github.com and run this script again."
    exit 1
  fi
fi

# 2. Log in if needed
if ! gh auth status >/dev/null 2>&1; then
  echo "Logging in to GitHub..."
  gh auth login
fi

# 3. Create the repo (private, with a README so it has a first commit)
if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  echo "Repo $OWNER/$REPO already exists - nothing to create."
else
  gh repo create "$OWNER/$REPO" --private --add-readme --description "$DESC"
  echo "Created private repo: https://github.com/$OWNER/$REPO"
fi

echo
echo "Last check: the Claude GitHub app needs access to this repo."
echo "  Open https://github.com/settings/installations, then Claude > Configure > Repository access."
echo "  If it says 'All repositories', you're done. Otherwise add '$REPO'."
