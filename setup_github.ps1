# PowerShell script to set up GitHub repository for LLM_FROM_SCRACT

Write-Host "Setting up GitHub repository for LLM_FROM_SCRACT..." -ForegroundColor Green

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "You can download it from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Or run: winget install --id Git.Git -e --source winget" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Initialize Git repository
Write-Host "Initializing Git repository..." -ForegroundColor Yellow
git init

# Add all files
Write-Host "Adding files to repository..." -ForegroundColor Yellow
git add .

# Create initial commit
Write-Host "Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: LLM from scratch chapter 3 - Attention mechanisms"

# Ask for GitHub repository URL
Write-Host ""
$repoUrl = Read-Host "Please provide your GitHub repository URL (e.g., https://github.com/username/repo-name.git)"

# Add remote origin
Write-Host "Adding remote origin..." -ForegroundColor Yellow
git remote add origin $repoUrl

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git branch -M main
git push -u origin main

Write-Host ""
Write-Host "Repository setup complete!" -ForegroundColor Green
Write-Host "Your code has been pushed to GitHub at: $repoUrl" -ForegroundColor Green
Read-Host "Press Enter to exit"
