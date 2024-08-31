echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="D:\software\ANSYS Inc\v241\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "D:\software\ANSYS Inc\v241\fluent\ntbin\win64\tell.exe" ALI-DESKTOP 57976 CLEANUP_EXITING
timeout /t 1
"D:\software\ANSYS Inc\v241\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 22964) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 2520) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 6988) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 15128) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 22684) 
if /i "%LOCALHOST%"=="ALI-DESKTOP" (%KILL_CMD% 21216)
del "C:\Users\Dell\OneDrive\Desktop\HPC Workshop\Ablation Modelling\cleanup-fluent-ALI-DESKTOP-22684.bat"
