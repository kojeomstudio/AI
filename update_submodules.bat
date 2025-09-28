@echo off
setlocal enabledelayedexpansion

REM This script updates all git submodules by fetching from 'upstream',
REM merging into the local branch, and pushing to 'origin'.
REM It then commits and pushes the updated submodule references in the parent repository.
REM This script assumes the branch to be updated is 'main'.

echo >>> Starting submodule update process...

REM Update each submodule
git submodule foreach --recursive "cmd /c ( ^
  echo ">>> Processing submodule: %name%" ^&^& ^
  echo ">>> Fetching from origin..." ^&^& ^
  git fetch origin ^&^& ^
  echo ">>> Checking out origin/main..." ^&^& ^
  git checkout origin/main ^
)"
IF %ERRORLEVEL% NEQ 0 (
    echo An error occurred while updating submodules.
    exit /b %ERRORLEVEL%
)

echo >>> Submodule updates complete.
echo >>> Committing and pushing changes in the parent repository...

REM Stage the updated submodule references
git add .

REM Commit the changes
git diff-index --quiet HEAD --
IF %ERRORLEVEL% EQU 0 (
  echo >>> No submodule changes to commit.
) ELSE (
  echo >>> Committing submodule updates...
  git commit -m "chore: Update submodules from upstream"
  IF %ERRORLEVEL% NEQ 0 (
    echo An error occurred while committing.
    exit /b %ERRORLEVEL%
  )
  
  REM Push the changes to the parent repository's main branch
  echo >>> Pushing parent repository to origin...
  git push origin main
  IF %ERRORLEVEL% NEQ 0 (
    echo An error occurred while pushing the parent repository.
    exit /b %ERRORLEVEL%
  )
)

echo >>> All done.
