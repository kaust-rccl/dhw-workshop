echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="D:\software\ANSYS Inc\v241\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "D:\software\ANSYS Inc\v241\fluent\ntbin\win64\tell.exe" ALI-DESKTOP 55486 CLEANUP_EXITING
timeout /t 1
"D:\software\ANSYS Inc\v241\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 23304) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 24444) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 1988) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 8848) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 23228) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 2568)
del "D:\HPC Workshop\Ansys Cases\parametric_mixer\parametric_ansys_project\cleanup-fluent-ALI-DESKTOP-23228.bat"
