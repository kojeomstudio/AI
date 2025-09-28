# This script updates all git submodules by fetching from 'upstream',
# merging into the local branch, and pushing to 'origin'.
# It then commits and pushes the updated submodule references in the parent repository.

Write-Host ">>> Starting submodule update process..."

# Update each submodule
try {
    git submodule foreach --recursive '
        $ErrorActionPreference = "Stop"
        Write-Host ">>> Processing submodule: $name"

        # Get the current branch name
        $branch = git rev-parse --abbrev-ref HEAD
        Write-Host ">>> On branch: $branch"

        # Fetch from upstream
        Write-Host ">>> Fetching from upstream..."
        git fetch upstream

        # Merge the corresponding upstream branch
        Write-Host ">>> Merging upstream/$branch into $branch..."
        git merge "upstream/$branch"

        # Push to origin
        Write-Host ">>> Pushing to origin..."
        git push origin "$branch"
    '
}
catch {
    Write-Error "An error occurred while updating submodules. Please check the output above."
    exit 1
}

Write-Host ">>> Submodule updates complete."
Write-Host ">>> Committing and pushing changes in the parent repository..."

# Stage the updated submodule references
git add .

# Commit the changes
# Check if there are staged changes before committing
$changes = git diff-index --quiet HEAD --
if ($LASTEXITCODE -eq 0) {
    Write-Host ">>> No submodule changes to commit."
}
else {
    Write-Host ">>> Committing submodule updates..."
    git commit -m "chore: Update submodules from upstream"

    # Push the changes to the parent repository's main branch
    Write-Host ">>> Pushing parent repository to origin..."
    git push origin main # Or your default branch
}

Write-Host ">>> All done."
