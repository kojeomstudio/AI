# This script resiliently updates all git submodules, logging any errors without halting.

$logFile = "submodule_update.log"
# Start with a clean log file for this run
if (Test-Path $logFile) {
    Clear-Content -Path $logFile
}
Add-Content -Path $logFile -Value "Submodule update process started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Write-Host ">>> Starting resilient submodule update process... Errors will be logged to '$logFile'"

# This script block contains the logic to be executed in each submodule.
$updateLogic = {
    param($submoduleName, $logFile)

    try {
        $ErrorActionPreference = 'Stop'
        Write-Host "`n>>> Processing submodule: $submoduleName"
        Write-Host ">>> Attempting 'upstream -> origin' workflow..."

        # 1. Check for detached HEAD and attempt to checkout a branch
        $branch = git rev-parse --abbrev-ref HEAD
        if ($branch -eq 'HEAD') {
            Write-Host "--- Submodule is in detached HEAD state. Attempting to checkout 'main' branch." -ForegroundColor Yellow
            git checkout main
            $branch = git rev-parse --abbrev-ref HEAD # Re-check branch after checkout
            if ($branch -eq 'HEAD') {
                # If still in detached HEAD, we cannot proceed with this workflow.
                throw "Cannot proceed: Submodule is in a detached HEAD state and 'main' branch could not be checked out."
            }
        }
        Write-Host ">>> On branch: $branch"

        # 2. Fetch from upstream
        Write-Host ">>> Fetching from upstream..."
        git fetch upstream

        # 3. Merge
        Write-Host ">>> Merging upstream/$branch into $branch..."
        git merge "upstream/$branch"

        # 4. Push to origin
        Write-Host ">>> Pushing to origin..."
        git push origin "$branch"

        Write-Host ">>> Successfully updated $submoduleName." -ForegroundColor Green
    }
    catch {
        # If any of the above commands fail, this block will execute.
        $errorMessage = "Failed in submodule '$submoduleName': $($_.Exception.Message)".Replace("`n"," ").Replace("`r"," ")
        
        # Log to console
        Write-Host $errorMessage -ForegroundColor Red

        # Log to file
        Add-Content -Path $logFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $errorMessage"
    }
}

try {
    # Get a list of all submodule paths from git.
    $submodulePaths = git submodule --quiet foreach --recursive 'echo $path'

    # Loop through each path in PowerShell.
    foreach ($path in $submodulePaths) {
        # Temporarily change to the submodule's directory.
        Push-Location $path
        
        # Execute the update logic, passing necessary parameters.
        & $updateLogic -submoduleName $path -logFile (Resolve-Path -Path $logFile -Relative)
        
        # Return to the original directory.
        Pop-Location
    }

    Write-Host "`n>>> Submodule update process complete."
    Write-Host ">>> Committing and pushing changes in the parent repository..."

    # Stage the updated submodule references.
    git add .

    # Commit the changes, but only if there are changes to commit.
    $changes = git diff-index --quiet HEAD --
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> Committing submodule updates..."
        git commit -m "chore: Update submodules from upstream (resilient)"
        
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
    # This will only catch critical errors like the submodule path not existing.
    Write-Error "A critical error occurred during the update process: $_"
    while(Get-Location -Stack) { Pop-Location }
    exit 1
}
