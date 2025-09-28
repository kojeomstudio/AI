# This script updates all git submodules using a robust, PowerShell-native approach.

Write-Host ">>> Starting submodule update process..."

# Define the script block to be executed in each submodule directory.
# This avoids complex quoting issues with `git submodule foreach`.
$updateLogic = {
    param($submoduleName)

    try {
        $ErrorActionPreference = 'Stop'
        Write-Host "`n>>> Processing submodule: $submoduleName"

        $branch = git rev-parse --abbrev-ref HEAD
        Write-Host ">>> On branch: $branch"

        Write-Host ">>> Fetching from upstream..."
        git fetch upstream

        Write-Host ">>> Merging upstream/$branch into $branch..."
        git merge "upstream/$branch"

        Write-Host ">>> Pushing to origin..."
        git push origin "$branch"
    }
    catch {
        # Throw a terminating error to be caught by the outer catch block.
        throw "Failed in submodule '$submoduleName': $_"
    }
}

try {
    # Get a list of all submodule paths from git by explicitly echoing the path.
    $submodulePaths = git submodule --quiet foreach --recursive 'echo $path'

    # Loop through each path in PowerShell.
    foreach ($path in $submodulePaths) {
        # Temporarily change to the submodule's directory.
        Push-Location $path
        
        # Execute the update logic defined above.
        & $updateLogic -submoduleName $path
        
        # Return to the original directory.
        Pop-Location
    }

    Write-Host "`n>>> Submodule updates complete."
    Write-Host ">>> Committing and pushing changes in the parent repository..."

    # Stage the updated submodule references.
    git add .

    # Commit the changes, but only if there are changes to commit.
    $changes = git diff-index --quiet HEAD --
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> Committing submodule updates..."
        git commit -m "chore: Update submodules from upstream"

        # Push the changes to the parent repository's main branch.
        Write-Host ">>> Pushing parent repository to origin..."
        git push origin main # Or your default branch
    }
    else {
        Write-Host ">>> No submodule changes to commit."
    }

    Write-Host ">>> All done."
}
catch {
    # If any submodule fails, the script will stop here.
    Write-Error "A submodule script failed. Halting execution. Error: $_"
    # Ensure we are back in the original directory in case of failure.
    while ($Host.UI.RawUI.KeyAvailable -and ($Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown").VirtualKeyCode -ne 13)) {}
    while(Get-Location -Stack) { Pop-Location }
    exit 1
}