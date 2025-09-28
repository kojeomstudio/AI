#!/bin/bash
# This script updates all git submodules by fetching from 'upstream',
# merging into the local branch, and pushing to 'origin'.
# It then commits and pushes the updated submodule references in the parent repository.

set -e # Exit immediately if a command exits with a non-zero status.

echo ">>> Starting submodule update process..."

# Update each submodule
git submodule foreach --recursive '
  echo ">>> Processing submodule: $name"
  
  # Get the current branch name
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  echo ">>> On branch: $BRANCH"
  
  # Fetch from upstream
  echo ">>> Fetching from upstream..."
  git fetch upstream
  
  # Merge the corresponding upstream branch
  echo ">>> Merging upstream/$BRANCH into $BRANCH..."
  git merge "upstream/$BRANCH"
  
  # Push to origin
  echo ">>> Pushing to origin..."
  git push origin "$BRANCH"
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
