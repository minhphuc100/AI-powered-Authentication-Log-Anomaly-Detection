#define MyAppName GetCmdParam("MyAppName", "AuthAnomalyDesktopApp")
#define MyAppVersion GetCmdParam("MyAppVersion", "1.0.0")
#define MyAppPublisher "AIC Project"
#define MyAppExeName "AuthAnomalyDesktopApp.exe"

[Setup]
AppId={{A7E7AF52-2F71-4D64-BB0C-7A11D7F4AF60}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=output
OutputBaseFilename=AuthAnomalyDesktopApp_Setup_{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\build\dist\AuthAnomalyDesktopApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{localappdata}\AuthAnomalyDesktopApp"
Name: "{localappdata}\AuthAnomalyDesktopApp\models"
Name: "{localappdata}\AuthAnomalyDesktopApp\exports"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
