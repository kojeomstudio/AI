#!/bin/bash
# This script updates all git submodules by fetching from 'upstream',
# merging into the local branch, and pushing to 'origin'.
# It then commits and pushes the updated submodule references in the parent repository.

set -e # Exit immediately if a command exits with a non-zero status.

echo ">>> Starting submodule update process..."

# Update each submodule
git submodule foreach --recursive '
  set -e
  echo ">>> Processing submodule: $name"
  
  echo ">>> Fetching from origin..."
  git fetch origin

  echo ">>> Finding default branch..."
  DEFAULT_BRANCH=$(git remote show origin | grep 'HEAD branch' | awk '{print $3}')
  if [ -z "$DEFAULT_BRANCH" ]; then
    echo "Error: Could not determine default branch for $name. Skipping."
    exit 1
  fi
  echo ">>> Default branch is '$DEFAULT_BRANCH'. Checking out origin/$DEFAULT_BRANCH..."
  git checkout "origin/$DEFAULT_BRANCH"
'

echo ">>> Submodule updates complete."
echo ">>> Committing and pushing changes in the parent repository..."

# Stage the updated submodule references
git add .

# Commit the changes
# Check if there are staged changes before committing
if git diff-index --quiet HEAD --; then
  echo ">>> No submodule changes to commit."
else
  echo ">>> Committing submodule updates..."
  git commit -m "chore: Update submodules from upstream"
  
  # Push the changes to the parent repository's main branch
  echo ">>> Pushing parent repository to origin..."
  git push origin main # Or your default branch
fi

echo ">>> All done."
