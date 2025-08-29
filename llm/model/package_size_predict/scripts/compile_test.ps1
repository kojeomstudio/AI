Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "Compiling Python files to bytecode (syntax check)"

$files = Get-ChildItem -Recurse -Filter *.py | Select-Object -ExpandProperty FullName
foreach ($f in $files) {
  python -m py_compile $f
}

Write-Host "Bytecode compilation succeeded for" $files.Count "files."

