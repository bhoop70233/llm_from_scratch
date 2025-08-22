@echo off
echo Setting up GitHub repository for LLM_FROM_SCRACT...

REM Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git is not installed. Please install Git first.
    echo You can download it from: https://git-scm.com/download/win
    echo Or run: winget install --id Git.Git -e --source winget
    pause
    exit /b 1
)

REM Initialize Git repository
echo Initializing Git repository...
git init

REM Add all files
echo Adding files to repository...
git add .

REM Create initial commit
echo Creating initial commit...
git commit -m "Initial commit: LLM from scratch chapter 3 - Attention mechanisms"

REM Ask for GitHub repository URL
echo.
echo Please provide your GitHub repository URL (e.g., https://github.com/username/repo-name.git):
set /p repo_url="GitHub URL: "

REM Add remote origin
echo Adding remote origin...
git remote add origin %repo_url%

REM Push to GitHub
echo Pushing to GitHub...
git branch -M main
git push -u origin main

echo.
echo Repository setup complete!
echo Your code has been pushed to GitHub at: %repo_url%
pause
