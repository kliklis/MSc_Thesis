@echo off
cd C:\path\to\your\repository
git add .
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set date=%%a-%%b-%%c
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set time=%%a-%%b
git commit -m "%date% %time%"
git push
pause
