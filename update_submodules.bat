@echo off
setlocal

set LOG_FILE=submodule_update.log

rem Start with a clean log file for this run
echo Submodule update process started at %date% %time% > %LOG_FILE%

echo >>> Starting resilient submodule update process... Errors will be logged to %LOG_FILE%

rem Use a simple for loop to get paths, as it's more robust than `git submodule foreach` for complex logic
for /f %%p in ('git submodule --quiet foreach --recursive "echo %%p" ') do (
    echo.
    echo >>> Processing submodule: %%p
    
    rem Change to the submodule directory
    pushd %%p

    rem Try the update logic. If any command fails, the `||` part will execute.
    ( 
        echo --- Attempting to checkout 'main' branch...
        git checkout main && (
            echo --- On branch 'main'. Attempting 'upstream -> origin' workflow...
            git fetch upstream && git merge upstream/main && git push origin main
        )
    ) || (
        echo Failed in submodule: %%p. See %LOG_FILE% for details.
        echo %date% %time% - Failed in submodule: %%p >> %LOG_FILE%
    )

    rem Return to the parent directory
    popd
)

echo.
echo >>> Submodule update process complete.
echo >>> Committing and pushing changes in the parent repository...

rem Stage the updated submodule references
git add .

rem Commit the changes, but only if there are changes to commit.
git diff-index --quiet HEAD --
if %ERRORLEVEL% neq 0 (
    echo >>> Committing submodule updates...
    git commit -m "chore: Update submodules from upstream (resilient)"
    
    rem Push the changes to the parent repository's main branch
    echo >>> Pushing parent repository to origin...
    git push origin main
) else (
    echo >>> No submodule changes to commit.
)

echo >>> All done.