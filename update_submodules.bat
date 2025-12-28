@echo off
setlocal EnableDelayedExpansion

rem This script resiliently updates all git submodules, logging any errors without halting.
rem Improvements:
rem   - Automatically detects the default branch (main/master) for each submodule
rem   - Handles cases where upstream remote doesn't exist (origin-only update)
rem   - Better error handling and logging

set LOG_FILE=submodule_update.log

rem Use a stable timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set "timestamp=%%I"
echo Submodule update process started at %timestamp% > %LOG_FILE%
echo ========================================== >> %LOG_FILE%

echo ^>^>^> Starting resilient submodule update process...
echo ^>^>^> Errors will be logged to %LOG_FILE%

rem Main loop for each submodule
for /f "usebackq delims=" %%p in (`git submodule --quiet foreach --recursive "echo %%p"`) do (
    echo.
    echo ^>^>^> Processing submodule: %%p
    pushd %%p
    call :update_submodule "%%p"
    popd
)

echo.
echo ^>^>^> Submodule update process complete.
echo ^>^>^> Committing and pushing changes in the parent repository...

rem Stage, commit, and push if there are changes
git add .
git diff-index --quiet HEAD --
if %ERRORLEVEL% neq 0 (
    echo ^>^>^> Committing submodule updates...
    git commit -m "chore: Update submodules from upstream (resilient)"

    rem Get current branch
    for /f %%b in ('git rev-parse --abbrev-ref HEAD') do set "parent_branch=%%b"
    echo ^>^>^> Pushing parent repository to origin/!parent_branch!...
    git push origin !parent_branch!
) else (
    echo ^>^>^> No submodule changes to commit.
)

echo.
echo ^>^>^> All done.
echo ^>^>^> Check '%LOG_FILE%' for any errors that occurred.
goto :eof

rem ==================================================================
:update_submodule
    set "submodule_path=%~1"

    rem Check current branch
    for /f %%b in ('git rev-parse --abbrev-ref HEAD') do set "current_branch=%%b"

    rem Handle detached HEAD
    if "!current_branch!"=="HEAD" (
        echo --- Submodule is in detached HEAD state.

        rem Try to detect and checkout the right branch
        set "target_branch="

        rem Check if upstream exists
        git remote get-url upstream >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            git fetch upstream --quiet 2>nul
            call :detect_branch upstream target_branch
        )

        if "!target_branch!"=="" (
            git fetch origin --quiet 2>nul
            call :detect_branch origin target_branch
        )

        if "!target_branch!"=="" set "target_branch=main"

        echo --- Attempting to checkout '!target_branch!' branch...
        git checkout !target_branch! 2>nul
        if !ERRORLEVEL! neq 0 (
            git checkout -b !target_branch! origin/!target_branch! 2>nul
            if !ERRORLEVEL! neq 0 (
                rem Try master as fallback
                git checkout master 2>nul
                if !ERRORLEVEL! equ 0 (
                    set "target_branch=master"
                ) else (
                    git checkout -b master origin/master 2>nul
                    if !ERRORLEVEL! equ 0 (
                        set "target_branch=master"
                    ) else (
                        call :log_error "!submodule_path!" "Cannot checkout any valid branch."
                        goto :eof
                    )
                )
            )
        )
        set "current_branch=!target_branch!"
    )

    echo ^>^>^> On branch: !current_branch!

    rem Check if upstream remote exists
    git remote get-url upstream >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo ^>^>^> Workflow: upstream -^> local -^> origin

        echo ^>^>^> Fetching from upstream...
        git fetch upstream
        if !ERRORLEVEL! neq 0 (
            call :log_error "!submodule_path!" "Failed to fetch from upstream."
            goto :eof
        )

        rem Check if current branch exists on upstream
        set "upstream_branch=!current_branch!"
        git show-ref --verify --quiet "refs/remotes/upstream/!upstream_branch!" 2>nul
        if !ERRORLEVEL! neq 0 (
            call :detect_branch upstream upstream_branch
            echo --- Current branch '!current_branch!' not found on upstream. Using upstream/!upstream_branch!
        )

        if not "!upstream_branch!"=="" (
            echo ^>^>^> Merging upstream/!upstream_branch! into !current_branch!...
            git merge "upstream/!upstream_branch!" --no-edit
            if !ERRORLEVEL! neq 0 (
                echo --- Merge conflict or error. Attempting to abort merge...
                git merge --abort 2>nul
                call :log_error "!submodule_path!" "Failed to merge from upstream."
                goto :eof
            )
        ) else (
            echo --- Warning: Could not determine upstream branch. Skipping merge.
        )

        echo ^>^>^> Pushing to origin...
        git push origin !current_branch!
        if !ERRORLEVEL! neq 0 (
            call :log_error "!submodule_path!" "Failed to push to origin."
            goto :eof
        )
    ) else (
        echo ^>^>^> Workflow: origin-only ^(no upstream remote^)

        echo ^>^>^> Fetching from origin...
        git fetch origin

        echo ^>^>^> Pulling from origin/!current_branch!...
        git pull origin !current_branch! --no-edit 2>nul
    )

    echo ^>^>^> Successfully updated !submodule_path!
goto :eof

rem ==================================================================
:detect_branch
    set "remote_name=%~1"
    set "result_var=%~2"

    rem Try to find main or master branch
    git show-ref --verify --quiet "refs/remotes/%remote_name%/main" 2>nul
    if !ERRORLEVEL! equ 0 (
        set "%result_var%=main"
        goto :eof
    )

    git show-ref --verify --quiet "refs/remotes/%remote_name%/master" 2>nul
    if !ERRORLEVEL! equ 0 (
        set "%result_var%=master"
        goto :eof
    )

    set "%result_var%="
goto :eof

rem ==================================================================
:log_error
    set "failed_path=%~1"
    set "error_message=%~2"
    echo Failed in submodule: %failed_path% - %error_message%
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set "timestamp=%%I"
    echo %timestamp% - Failed in submodule: %failed_path% - %error_message% >> %LOG_FILE%
    echo --- >> %LOG_FILE%
goto :eof
