# This script resiliently updates all git submodules, logging any errors without halting.
# Improvements:
#   - Automatically detects the default branch (main/master) for each submodule
#   - Handles cases where upstream remote doesn't exist (origin-only update)
#   - Better error handling and logging

$logFile = "submodule_update.log"

# Start with a clean log file for this run
if (Test-Path $logFile) {
    Clear-Content -Path $logFile
}
Add-Content -Path $logFile -Value "Submodule update process started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Add-Content -Path $logFile -Value "=========================================="

Write-Host ">>> Starting resilient submodule update process..."
Write-Host ">>> Errors will be logged to '$logFile'"

# Function to detect default branch from a remote
function Get-DefaultBranch {
    param($remote)

    $defaultBranch = $null

    # Try to get the default branch from remote HEAD
    try {
        $remoteInfo = git remote show $remote 2>&1
        $headLine = $remoteInfo | Where-Object { $_ -match 'HEAD branch' }
        if ($headLine) {
            $defaultBranch = ($headLine -split ':')[-1].Trim()
        }
    } catch { }

    if ([string]::IsNullOrEmpty($defaultBranch) -or $defaultBranch -eq '(unknown)') {
        # Fallback: check if main or master exists
        $mainExists = git show-ref --verify --quiet "refs/remotes/$remote/main" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $defaultBranch = "main"
        } else {
            $masterExists = git show-ref --verify --quiet "refs/remotes/$remote/master" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $defaultBranch = "master"
            }
        }
    }

    return $defaultBranch
}

# Function to check if a remote exists
function Test-Remote {
    param($remoteName)
    $null = git remote get-url $remoteName 2>&1
    return $LASTEXITCODE -eq 0
}

# This script block contains the logic to be executed in each submodule.
$updateLogic = {
    param($submoduleName, $logFile)

    try {
        $ErrorActionPreference = 'Stop'
        Write-Host "`n>>> Processing submodule: $submoduleName"

        # Check for detached HEAD and attempt to checkout a branch
        $currentBranch = git rev-parse --abbrev-ref HEAD

        if ($currentBranch -eq 'HEAD') {
            Write-Host "--- Submodule is in detached HEAD state." -ForegroundColor Yellow

            # First, try to determine the appropriate branch
            $targetBranch = $null

            if (Test-Remote "upstream") {
                git fetch upstream --quiet 2>&1 | Out-Null
                $targetBranch = Get-DefaultBranch "upstream"
            }

            if ([string]::IsNullOrEmpty($targetBranch)) {
                git fetch origin --quiet 2>&1 | Out-Null
                $targetBranch = Get-DefaultBranch "origin"
            }

            if ([string]::IsNullOrEmpty($targetBranch)) {
                $targetBranch = "main"
            }

            Write-Host "--- Attempting to checkout '$targetBranch' branch..."

            $checkoutSuccess = $false
            git checkout $targetBranch 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $checkoutSuccess = $true
            } else {
                git checkout -b $targetBranch "origin/$targetBranch" 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    $checkoutSuccess = $true
                } else {
                    # Try master as fallback
                    git checkout master 2>&1 | Out-Null
                    if ($LASTEXITCODE -eq 0) {
                        $targetBranch = "master"
                        $checkoutSuccess = $true
                    } else {
                        git checkout -b master "origin/master" 2>&1 | Out-Null
                        if ($LASTEXITCODE -eq 0) {
                            $targetBranch = "master"
                            $checkoutSuccess = $true
                        }
                    }
                }
            }

            if (-not $checkoutSuccess) {
                throw "Cannot proceed: Unable to checkout any valid branch."
            }
            $currentBranch = $targetBranch
        }

        Write-Host ">>> On branch: $currentBranch"

        # Check if upstream remote exists
        if (Test-Remote "upstream") {
            Write-Host ">>> Workflow: upstream -> local -> origin"

            # Fetch from upstream
            Write-Host ">>> Fetching from upstream..."
            git fetch upstream

            # Detect upstream's default branch if current branch doesn't exist on upstream
            $upstreamBranch = $currentBranch
            $branchExists = git show-ref --verify --quiet "refs/remotes/upstream/$upstreamBranch" 2>&1
            if ($LASTEXITCODE -ne 0) {
                $upstreamBranch = Get-DefaultBranch "upstream"
                Write-Host "--- Current branch '$currentBranch' not found on upstream. Using upstream/$upstreamBranch" -ForegroundColor Yellow
            }

            if (-not [string]::IsNullOrEmpty($upstreamBranch)) {
                Write-Host ">>> Merging upstream/$upstreamBranch into $currentBranch..."
                $mergeResult = git merge "upstream/$upstreamBranch" --no-edit 2>&1
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "--- Merge conflict or error. Attempting to abort merge..." -ForegroundColor Yellow
                    git merge --abort 2>&1 | Out-Null
                    throw "Merge failed"
                }
            } else {
                Write-Host "--- Warning: Could not determine upstream branch. Skipping merge." -ForegroundColor Yellow
            }

            Write-Host ">>> Pushing to origin..."
            git push origin $currentBranch
        } else {
            Write-Host ">>> Workflow: origin-only (no upstream remote)"

            # Just fetch and pull from origin
            Write-Host ">>> Fetching from origin..."
            git fetch origin

            Write-Host ">>> Pulling from origin/$currentBranch..."
            git pull origin $currentBranch --no-edit 2>&1 | Out-Null
        }

        Write-Host ">>> Successfully updated $submoduleName." -ForegroundColor Green
    }
    catch {
        # If any of the above commands fail, this block will execute.
        $errorMessage = "Failed in submodule '$submoduleName': $($_.Exception.Message)".Replace("`n"," ").Replace("`r"," ")

        # Log to console
        Write-Host $errorMessage -ForegroundColor Red

        # Log to file
        Add-Content -Path $logFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $errorMessage"
        Add-Content -Path $logFile -Value "---"
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
    $null = git diff-index --quiet HEAD --
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> Committing submodule updates..."
        git commit -m "chore: Update submodules from upstream (resilient)"

        # Push the changes to the parent repository's current branch.
        $parentBranch = git rev-parse --abbrev-ref HEAD
        Write-Host ">>> Pushing parent repository to origin/$parentBranch..."
        git push origin $parentBranch
    }
    else {
        Write-Host ">>> No submodule changes to commit."
    }

    Write-Host "`n>>> All done."
    Write-Host ">>> Check '$logFile' for any errors that occurred."
}
catch {
    # This will only catch critical errors like the submodule path not existing.
    Write-Error "A critical error occurred during the update process: $_"
    while(Get-Location -Stack) { Pop-Location }
    exit 1
}
