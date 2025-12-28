#!/bin/bash
# This script resiliently updates all git submodules, logging any errors without halting.
# Improvements:
#   - Automatically detects the default branch (main/master) for each submodule
#   - Handles cases where upstream remote doesn't exist (origin-only update)
#   - Better error handling and logging

LOG_FILE="$(pwd)/submodule_update.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start with a clean log file for this run
echo "Submodule update process started at $(date)" > "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

echo ">>> Starting resilient submodule update process..."
echo ">>> Errors will be logged to '$LOG_FILE'"

# Function to detect default branch from a remote
detect_default_branch() {
  local remote=$1
  local default_branch=""

  # Try to get the default branch from remote HEAD
  default_branch=$(git remote show "$remote" 2>/dev/null | grep 'HEAD branch' | awk '{print $NF}')

  if [ -z "$default_branch" ] || [ "$default_branch" = "(unknown)" ]; then
    # Fallback: check if main or master exists
    if git show-ref --verify --quiet "refs/remotes/$remote/main" 2>/dev/null; then
      default_branch="main"
    elif git show-ref --verify --quiet "refs/remotes/$remote/master" 2>/dev/null; then
      default_branch="master"
    fi
  fi

  echo "$default_branch"
}

# Function to check if a remote exists
has_remote() {
  git remote get-url "$1" &>/dev/null
}

# Get a list of all submodule paths and loop through them
for path in $(git submodule --quiet foreach --recursive 'echo $path'); do
  # Use a subshell to isolate each submodule's operations
  (
    set -e # Exit subshell immediately on error
    cd "$path"

    echo ""
    echo ">>> Processing submodule: $path"

    # Check for detached HEAD and attempt to checkout a branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    if [ "$CURRENT_BRANCH" = "HEAD" ]; then
      echo "--- Submodule is in detached HEAD state."

      # First, try to determine the appropriate branch
      if has_remote "upstream"; then
        git fetch upstream --quiet
        TARGET_BRANCH=$(detect_default_branch "upstream")
      fi

      if [ -z "$TARGET_BRANCH" ]; then
        git fetch origin --quiet
        TARGET_BRANCH=$(detect_default_branch "origin")
      fi

      if [ -z "$TARGET_BRANCH" ]; then
        # Last resort: try main then master
        TARGET_BRANCH="main"
      fi

      echo "--- Attempting to checkout '$TARGET_BRANCH' branch..."
      git checkout "$TARGET_BRANCH" 2>/dev/null || git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH" 2>/dev/null || {
        git checkout master 2>/dev/null || git checkout -b master "origin/master" 2>/dev/null || {
          echo "Cannot proceed: Unable to checkout any valid branch."
          exit 1
        }
        TARGET_BRANCH="master"
      }
      CURRENT_BRANCH="$TARGET_BRANCH"
    fi

    echo ">>> On branch: $CURRENT_BRANCH"

    # Check if upstream remote exists
    if has_remote "upstream"; then
      echo ">>> Workflow: upstream -> local -> origin"

      # Fetch from upstream
      echo ">>> Fetching from upstream..."
      git fetch upstream

      # Detect upstream's default branch if current branch doesn't exist on upstream
      UPSTREAM_BRANCH="$CURRENT_BRANCH"
      if ! git show-ref --verify --quiet "refs/remotes/upstream/$UPSTREAM_BRANCH" 2>/dev/null; then
        UPSTREAM_BRANCH=$(detect_default_branch "upstream")
        echo "--- Current branch '$CURRENT_BRANCH' not found on upstream. Using upstream/$UPSTREAM_BRANCH"
      fi

      if [ -n "$UPSTREAM_BRANCH" ]; then
        echo ">>> Merging upstream/$UPSTREAM_BRANCH into $CURRENT_BRANCH..."
        # Use --no-edit to avoid opening editor for merge commit
        git merge "upstream/$UPSTREAM_BRANCH" --no-edit || {
          echo "--- Merge conflict or error. Attempting to abort merge..."
          git merge --abort 2>/dev/null || true
          exit 1
        }
      else
        echo "--- Warning: Could not determine upstream branch. Skipping merge."
      fi

      echo ">>> Pushing to origin..."
      git push origin "$CURRENT_BRANCH"
    else
      echo ">>> Workflow: origin-only (no upstream remote)"

      # Just fetch and pull from origin
      echo ">>> Fetching from origin..."
      git fetch origin

      echo ">>> Pulling from origin/$CURRENT_BRANCH..."
      git pull origin "$CURRENT_BRANCH" --no-edit 2>/dev/null || {
        echo "--- Pull failed. Branch may not exist on origin or other error."
      }
    fi

    echo ">>> Successfully updated $path"

  ) || {
    # If the subshell fails at any point, this block will execute.
    ERROR_MSG="$(date) - Failed in submodule: $path"
    echo "$ERROR_MSG" | tee -a "$LOG_FILE"
    echo "---" >> "$LOG_FILE"
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

  # Push the changes to the parent repository's default branch
  PARENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
  echo ">>> Pushing parent repository to origin/$PARENT_BRANCH..."
  git push origin "$PARENT_BRANCH"
else
  echo ">>> No submodule changes to commit."
fi

echo ""
echo ">>> All done."
echo ">>> Check '$LOG_FILE' for any errors that occurred."