$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildRoot = Join-Path $projectRoot 'build'
$distPath = Join-Path $buildRoot 'dist'
$workPath = Join-Path $buildRoot 'pyinstaller-work'
$specPath = Join-Path $buildRoot 'spec'
$installerScript = Join-Path $projectRoot 'installer\AuthAnomalyDesktopApp.iss'
$installerOutput = Join-Path $projectRoot 'installer\output'
$appName = 'AuthAnomalyDesktopApp'
$appVersion = '1.0.0'

function Resolve-PythonCommand {
    # Prefer the Windows Python launcher, which works even when python.exe is not
    # added to PATH. Fall back to normal command names for other installations.
    foreach ($name in @('py.exe', 'py', 'python.exe', 'python', 'python3.exe', 'python3')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    throw @'
Không tìm thấy Python 3. Hãy cài Python từ https://www.python.org/downloads/windows/
và chọn "Add python.exe to PATH", sau đó mở lại PowerShell và chạy lại script.
'@
}

function Invoke-Python {
    param([string[]]$PythonArguments)

    & $script:pythonCommand @script:pythonLauncherArguments @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Lệnh Python thất bại (exit code $LASTEXITCODE)."
    }
}

$pythonCommand = Resolve-PythonCommand
$pythonLauncherArguments = if ((Split-Path -Leaf $pythonCommand) -match '^py(\.exe)?$') { @('-3') } else { @() }

function Resolve-IsccPath {
    $candidates = @(
        'C:\Users\buinh\AppData\Local\Programs\Inno Setup 6\ISCC.exe',
        'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
        'C:\Program Files\Inno Setup 6\ISCC.exe'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $registryPaths = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1'
    )

    foreach ($registryPath in $registryPaths) {
        if (Test-Path $registryPath) {
            $installLocation = (Get-ItemProperty $registryPath).InstallLocation
            if ($installLocation) {
                $resolved = Join-Path $installLocation 'ISCC.exe'
                if (Test-Path $resolved) {
                    return $resolved
                }
            }
        }
    }

    throw 'Không tìm thấy ISCC.exe của Inno Setup.'
}

Write-Host '==> Cài dependencies dự án'
Invoke-Python @('-m', 'pip', 'install', '-r', (Join-Path $projectRoot 'requirements.txt'))
Invoke-Python @('-m', 'pip', 'install', 'pyinstaller')

Write-Host '==> Dọn build cũ'
foreach ($path in @($distPath, $workPath, $specPath, $installerOutput)) {
    if (Test-Path $path) {
        Remove-Item -Path $path -Recurse -Force
    }
}

New-Item -ItemType Directory -Path $distPath | Out-Null
New-Item -ItemType Directory -Path $workPath | Out-Null
New-Item -ItemType Directory -Path $specPath | Out-Null
New-Item -ItemType Directory -Path $installerOutput | Out-Null

Write-Host '==> Build executable bằng PyInstaller'
Invoke-Python @(
    '-m', 'PyInstaller',
    '--noconfirm',
    '--clean',
    '--windowed',
    '--name', $appName,
    '--distpath', $distPath,
    '--workpath', $workPath,
    '--specpath', $specPath,
    '--collect-all', 'PySide6',
    '--hidden-import', 'pywintypes',
    '--hidden-import', 'pythoncom',
    '--hidden-import', 'win32timezone',
    (Join-Path $projectRoot 'run_desktop_app.py')
)

$exePath = Join-Path $distPath "$appName\$appName.exe"
if (-not (Test-Path $exePath)) {
    throw "Không tìm thấy file exe sau khi build: $exePath"
}

Write-Host '==> Build setup.exe bằng Inno Setup'
$isccPath = Resolve-IsccPath
& $isccPath "/DMyAppVersion=$appVersion" "/DMyAppName=$appName" $installerScript

$setupFile = Get-ChildItem -Path $installerOutput -Filter '*.exe' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $setupFile) {
    throw 'Không tìm thấy file setup.exe sau khi build installer.'
}

Write-Host ''
Write-Host "Build hoàn tất:"
Write-Host "  EXE:   $exePath"
Write-Host "  SETUP: $($setupFile.FullName)"
