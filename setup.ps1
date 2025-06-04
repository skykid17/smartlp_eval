# PowerShell Setup Script for RAG Log Analysis System
# Run this script to set up the complete system

param(
    [switch]$SkipDependencyCheck,
    [switch]$ForceRefresh
)

# Color functions for better output
function Write-Header {
    param([string]$Title)
    Write-Host "`n$('='*60)" -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host "$('='*60)" -ForegroundColor Cyan
}

function Write-Step {
    param([int]$StepNum, [string]$Description)
    Write-Host "`n[$StepNum] $Description" -ForegroundColor Yellow
    Write-Host "$('-'*40)" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

# Function to check if a command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to check if a port is open
function Test-Port {
    param([string]$Host, [int]$Port, [int]$Timeout = 5)
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $connection = $tcpClient.BeginConnect($Host, $Port, $null, $null)
        $wait = $connection.AsyncWaitHandle.WaitOne($Timeout * 1000, $false)
        
        if ($wait) {
            $tcpClient.EndConnect($connection)
            $tcpClient.Close()
            return $true
        } else {
            $tcpClient.Close()
            return $false
        }
    } catch {
        return $false
    }
}

Write-Header "RAG Log Analysis System - Windows Setup"

Write-Host "This script will help you set up the RAG log analysis system."
Write-Host "Prerequisites that will be checked:"
Write-Host "- Python 3.8+"
Write-Host "- Git"
Write-Host "- Docker (for Milvus)"
Write-Host "- Ollama service"
Write-Host ""

if (-not $SkipDependencyCheck) {
    $continue = Read-Host "Continue with setup? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        Write-Host "Setup cancelled."
        exit 1
    }
}

$setupResults = @{}

# Step 1: Check Python
Write-Step 1 "Checking Python Installation"

if (Test-Command "python") {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -ge 3 -and ($major -gt 3 -or $minor -ge 8)) {
            Write-Success "Python $($matches[1]).$($matches[2]).$($matches[3]) found"
            $setupResults["Python"] = $true
        } else {
            Write-Error "Python 3.8+ required, found $($matches[1]).$($matches[2]).$($matches[3])"
            $setupResults["Python"] = $false
        }
    } else {
        Write-Error "Could not determine Python version"
        $setupResults["Python"] = $false
    }
} else {
    Write-Error "Python not found. Please install Python 3.8+ from https://python.org"
    $setupResults["Python"] = $false
}

# Step 2: Check Git
Write-Step 2 "Checking Git Installation"

if (Test-Command "git") {
    $gitVersion = git --version
    Write-Success "Git found: $gitVersion"
    $setupResults["Git"] = $true
} else {
    Write-Error "Git not found. Please install Git from https://git-scm.com"
    $setupResults["Git"] = $false
}

# Step 3: Check Docker
Write-Step 3 "Checking Docker Installation"

if (Test-Command "docker") {
    try {
        $dockerVersion = docker --version
        Write-Success "Docker found: $dockerVersion"
        
        # Check if Docker is running
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker daemon is running"
            $setupResults["Docker"] = $true
        } else {
            Write-Warning "Docker is installed but daemon is not running"
            Write-Host "Please start Docker Desktop"
            $setupResults["Docker"] = $false
        }
    } catch {
        Write-Error "Docker command failed"
        $setupResults["Docker"] = $false
    }
} else {
    Write-Error "Docker not found. Please install Docker Desktop"
    $setupResults["Docker"] = $false
}

# Step 4: Setup Virtual Environment
Write-Step 4 "Setting up Python Virtual Environment"

if ($setupResults["Python"]) {
    try {
        if (-not (Test-Path ".venv")) {
            Write-Host "Creating virtual environment..."
            python -m venv .venv
        }
        
        Write-Host "Activating virtual environment..."
        & ".venv\Scripts\Activate.ps1"
        
        Write-Host "Upgrading pip..."
        python -m pip install --upgrade pip
        
        Write-Success "Virtual environment ready"
        $setupResults["VirtualEnv"] = $true
    } catch {
        Write-Error "Failed to setup virtual environment: $_"
        $setupResults["VirtualEnv"] = $false
    }
} else {
    Write-Error "Skipping virtual environment setup (Python not available)"
    $setupResults["VirtualEnv"] = $false
}

# Step 5: Install Python Dependencies
Write-Step 5 "Installing Python Dependencies"

if ($setupResults["VirtualEnv"]) {
    try {
        Write-Host "Installing requirements..."
        python -m pip install -r requirements.txt
        Write-Success "Dependencies installed"
        $setupResults["Dependencies"] = $true
    } catch {
        Write-Error "Failed to install dependencies: $_"
        $setupResults["Dependencies"] = $false
    }
} else {
    Write-Error "Skipping dependencies (virtual environment not ready)"
    $setupResults["Dependencies"] = $false
}

# Step 6: Check/Start Milvus
Write-Step 6 "Setting up Milvus Vector Database"

if ($setupResults["Docker"]) {
    # Check if Milvus is already running
    if (Test-Port "localhost" 19530) {
        Write-Success "Milvus is already running on port 19530"
        $setupResults["Milvus"] = $true
    } else {
        Write-Host "Starting Milvus container..."
        try {
            docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
            
            Write-Host "Waiting for Milvus to start..."
            $attempts = 0
            $maxAttempts = 30
            
            do {
                Start-Sleep -Seconds 2
                $attempts++
                Write-Host "." -NoNewline
            } while (-not (Test-Port "localhost" 19530) -and $attempts -lt $maxAttempts)
            
            Write-Host ""
            
            if (Test-Port "localhost" 19530) {
                Write-Success "Milvus started successfully"
                $setupResults["Milvus"] = $true
            } else {
                Write-Error "Milvus failed to start within timeout"
                $setupResults["Milvus"] = $false
            }
        } catch {
            Write-Error "Failed to start Milvus: $_"
            $setupResults["Milvus"] = $false
        }
    }
} else {
    Write-Error "Skipping Milvus setup (Docker not available)"
    $setupResults["Milvus"] = $false
}

# Step 7: Check Ollama
Write-Step 7 "Checking Ollama Service"

if (Test-Command "ollama") {
    Write-Success "Ollama CLI found"
    
    # Check if Ollama service is running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
        $models = ($response.Content | ConvertFrom-Json).models
        
        Write-Success "Ollama service is running"
        Write-Host "Available models: $($models.Count)"
        
        # Check required models
        $requiredModels = @("llama3.2", "nomic-embed-text")
        $missingModels = @()
        
        foreach ($required in $requiredModels) {
            $found = $false
            foreach ($model in $models) {
                if ($model.name -like "*$required*") {
                    $found = $true
                    break
                }
            }
            if (-not $found) {
                $missingModels += $required
            }
        }
        
        if ($missingModels.Count -eq 0) {
            Write-Success "All required models are available"
            $setupResults["Ollama"] = $true
        } else {
            Write-Warning "Missing models: $($missingModels -join ', ')"
            Write-Host "Pulling missing models..."
            
            foreach ($model in $missingModels) {
                Write-Host "Pulling $model..."
                ollama pull $model
            }
            
            Write-Success "Models pulled successfully"
            $setupResults["Ollama"] = $true
        }
        
    } catch {
        Write-Error "Ollama service not running. Please start it with 'ollama serve'"
        $setupResults["Ollama"] = $false
    }
} else {
    Write-Error "Ollama not found. Please install from https://ollama.ai"
    $setupResults["Ollama"] = $false
}

# Step 8: Check Elastic Integrations
Write-Step 8 "Checking Elastic Integrations Repository"

$integrationsPath = "C:\Users\geola\Documents\GitHub\elastic_integrations"

if (Test-Path $integrationsPath) {
    $integrationCount = (Get-ChildItem $integrationsPath -Directory | Where-Object { -not $_.Name.StartsWith('.') }).Count
    Write-Success "Elastic integrations found at $integrationsPath"
    Write-Host "Integration packages: $integrationCount"
    $setupResults["Integrations"] = $true
} else {
    Write-Warning "Elastic integrations repository not found"
    Write-Host "Cloning Elastic integrations repository..."
    
    try {
        $parentDir = Split-Path $integrationsPath -Parent
        if (-not (Test-Path $parentDir)) {
            New-Item -ItemType Directory -Path $parentDir -Force
        }
        
        git clone https://github.com/elastic/integrations.git $integrationsPath
        
        if (Test-Path $integrationsPath) {
            Write-Success "Elastic integrations cloned successfully"
            $setupResults["Integrations"] = $true
        } else {
            Write-Error "Failed to clone integrations repository"
            $setupResults["Integrations"] = $false
        }
    } catch {
        Write-Error "Failed to clone integrations: $_"
        $setupResults["Integrations"] = $false
    }
}

# Step 9: Initialize System Data
Write-Step 9 "Initializing System Data"

$canInitialize = $setupResults["Dependencies"] -and $setupResults["Ollama"] -and $setupResults["Milvus"] -and $setupResults["Integrations"]

if ($canInitialize) {
    try {
        Write-Host "Running system initialization..."
        if ($ForceRefresh) {
            python setup.py
        } else {
            python -c "from data_initialization import initialize_integration_data; print('Result:', initialize_integration_data())"
        }
        
        Write-Success "System data initialized"
        $setupResults["SystemData"] = $true
    } catch {
        Write-Error "Failed to initialize system data: $_"
        $setupResults["SystemData"] = $false
    }
} else {
    Write-Error "Skipping system initialization (dependencies not met)"
    $setupResults["SystemData"] = $false
}

# Summary
Write-Header "Setup Summary"

$successCount = ($setupResults.Values | Where-Object { $_ -eq $true }).Count
$totalSteps = $setupResults.Count

Write-Host "Completed: $successCount/$totalSteps steps" -ForegroundColor Cyan
Write-Host "`nStep Results:" -ForegroundColor Cyan

foreach ($step in $setupResults.GetEnumerator()) {
    $status = if ($step.Value) { "✅ PASS" } else { "❌ FAIL" }
    Write-Host "   $($step.Key): $status"
}

if ($successCount -eq $totalSteps) {
    Write-Header "🎉 Setup Completed Successfully!"
    
    Write-Host "`nNext steps:" -ForegroundColor Green
    Write-Host "   1. Run: python cli.py validate" -ForegroundColor White
    Write-Host "   2. Run: python cli.py test" -ForegroundColor White
    Write-Host "   3. Run: python example_usage.py" -ForegroundColor White
    Write-Host "   4. Try: python cli.py analyze -f your_log_file.log" -ForegroundColor White
    
    Write-Host "`nSystem is ready to use! 🚀" -ForegroundColor Green
    exit 0
} else {
    Write-Header "⚠️  Setup Completed with Issues"
    
    Write-Host "`nPlease resolve the failed steps before using the system." -ForegroundColor Yellow
    Write-Host "You can re-run this script after fixing the issues." -ForegroundColor Yellow
    
    exit 1
}
