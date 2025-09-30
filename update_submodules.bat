@echo off
setlocal

set LOG_FILE=submodule_update.log

rem Use a stable timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set "timestamp=%%I"
echo Submodule update process started at %timestamp% > %LOG_FILE%

echo >>> Starting resilient submodule update process... Errors will be logged to %LOG_FILE%

rem Main loop for each submodule
for /f %%p in ('git submodule --quiet foreach --recursive "echo %%p"') do (
    echo.
    echo >>> Processing submodule: %%p
    pushd %%p
    call :update_submodule %%p
    popd
)

echo.
echo >>> Submodule update process complete.
echo >>> Committing and pushing changes in the parent repository...

rem Stage, commit, and push if there are changes
git add .
git diff-index --quiet HEAD --
if %ERRORLEVEL% neq 0 (
    echo >>> Committing submodule updates...
    git commit -m "chore: Update submodules from upstream (resilient)"
    echo >>> Pushing parent repository to origin...
    git push origin main
) else (
    echo >>> No submodule changes to commit.
)

echo >>> All done.
goto :eof

rem ==================================================================
:update_submodule
    rem This is the logic block for a single submodule
    set "submodule_path=%~1"

    rem Try to checkout main and run the workflow
    git checkout main
    if %ERRORLEVEL% neq 0 (
        call :log_error %submodule_path% "Failed to checkout main branch."
        goto :eof
    )

    git fetch upstream
    if %ERRORLEVEL% neq 0 (
        call :log_error %submodule_path% "Failed to fetch from upstream."
        goto :eof
    )

    git merge upstream/main
    if %ERRORLEVEL% neq 0 (
        call :log_error %submodule_path% "Failed to merge from upstream/main."
        goto :eof
    )

    git push origin main
    if %ERRORLEVEL% neq 0 (
        call :log_error %submodule_path% "Failed to push to origin."
        goto :eof
    )

    echo >>> Successfully updated %submodule_path%
goto :eof

rem ==================================================================
:log_error
    set "failed_path=%~1"
    set "error_message=%~2"
    echo Failed in submodule: %failed_path% - %error_message%
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set "timestamp=%%I"
    echo %timestamp% - Failed in submodule: %failed_path% - %error_message% >> %LOG_FILE%
goto :eof
