chcp 65001
@echo off
setlocal

:: Git root ?붾젆?좊━ 李얘린
for /f "delims=" %%i in ('git rev-parse --show-toplevel') do set GIT_ROOT=%%i

if "%GIT_ROOT%"=="" (
    echo Git root瑜?李얠쓣 ???놁뒿?덈떎.
    exit /b 1
)

echo Git root: %GIT_ROOT%
cd /d "%GIT_ROOT%"

echo ?쒕툕紐⑤뱢 珥덇린??諛??낅뜲?댄듃瑜??쒖옉?⑸땲??.. (Recursive)
git submodule update --init --recursive

echo ?쒕툕紐⑤뱢 ?낅뜲?댄듃媛 ?꾨즺?섏뿀?듬땲??
pause
