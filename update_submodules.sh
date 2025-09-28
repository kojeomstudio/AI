#!/bin/bash
# This script resiliently updates all git submodules, logging any errors without halting.

LOG_FILE="$(pwd)/submodule_update.log"

# Start with a clean log file for this run
_="$(echo "Submodule update process started at $(date)" > "$LOG_FILE")"

echo ">>> Starting resilient submodule update process... Errors will be logged to '$LOG_FILE'"

# Get a list of all submodule paths and loop through them
for path in $(git submodule --quiet foreach --recursive 'echo $path'); do
  # Use a subshell to isolate each submodule's operations
  ( 
    set -e # Exit subshell immediately on error
    cd "$path"

    echo ""
    echo ">>> Processing submodule: $path"
    echo ">>> Attempting 'upstream -> origin' workflow..."

    # Check for detached HEAD and attempt to checkout a branch
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$BRANCH" = "HEAD" ]; then
      echo "--- Submodule is in detached HEAD state. Attempting to checkout 'main' branch."
      git checkout main
      BRANCH=$(git rev-parse --abbrev-ref HEAD)
      if [ "$BRANCH" = "HEAD" ]; then
        echo "Cannot proceed: Still in detached HEAD after trying to checkout main."
        exit 1 # Exit subshell
      fi
    fi
    echo ">>> On branch: $BRANCH"

    echo ">>> Fetching from upstream..."
    git fetch upstream

    echo ">>> Merging upstream/$BRANCH into $BRANCH..."
    git merge "upstream/$BRANCH"

    echo ">>> Pushing to origin..."
    git push origin "$BRANCH"

    echo ">>> Successfully updated $path"

  ) || { 
    # If the subshell fails at any point, this block will execute.
    ERROR_MSG="$(date) - Failed in submodule: $path"
    echo "$ERROR_MSG" | tee -a "$LOG_FILE"
  }
done

echo ""
echo ">>> Submodule update process complete."
echo ">>> Committing and pushing changes in the parent repository..."

# Stage the updated submodule references
git add .

# Commit the changes, but only if there are changes to commit.
if ! git diff-index --quiet HEAD --; then
  echo ">>> Committing submodule updates..."
  git commit -m "chore: Update submodules from upstream (resilient)"
  
  # Push the changes to the parent repository's main branch
  echo ">>> Pushing parent repository to origin..."
  git push origin main # Or your default branch
else
  echo ">>> No submodule changes to commit."
fi

echo ">>> All done."