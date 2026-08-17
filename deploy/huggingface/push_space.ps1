# Publish the API to a Hugging Face Space (Docker SDK).
#
#   powershell -ExecutionPolicy Bypass -File deploy\huggingface\push_space.ps1 `
#       -User <hf-username> -Space voicerag
#
# Prompts for an HF token with *write* scope (https://huggingface.co/settings/tokens)
# without echoing it. Builds a clean worktree containing exactly what the Space
# needs, and pushes it to the Space's git remote. The Space then builds the
# Docker image itself -- including the index, from the organisers' dataset.
#
# Secrets (SARVAM_API_KEY, GROQ_API_KEY, OPENAI_API_KEY) are set in the Space UI
# under Settings -> Variables and secrets. They are deliberately NOT pushed.

param(
    [Parameter(Mandatory = $true)][string]$User,
    [string]$Space = "voicerag",
    [string]$IndexRows = "20000"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$staging = Join-Path $env:TEMP "voicerag-space"

function Read-Secret([string]$Prompt) {
    $secure = Read-Host -Prompt $Prompt -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

Write-Host "Hugging Face token needs *write* scope: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
$token = Read-Secret "HF_TOKEN"
if ([string]::IsNullOrWhiteSpace($token)) { Write-Error "no token given"; exit 1 }

# --- build a clean staging tree ------------------------------------------------
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory -Path $staging | Out-Null

foreach ($d in @("src", "eval", "scripts")) {
    Copy-Item -Recurse (Join-Path $repo $d) (Join-Path $staging $d)
}
Copy-Item (Join-Path $repo "requirements.txt") $staging
Copy-Item (Join-Path $repo "pyproject.toml") $staging
Copy-Item (Join-Path $repo ".dockerignore") $staging
# The Space expects its Dockerfile at the repo root, and its README carries the
# YAML frontmatter that tells HF this is a Docker Space listening on 8000.
Copy-Item (Join-Path $repo "deploy\Dockerfile") (Join-Path $staging "Dockerfile")
Copy-Item (Join-Path $PSScriptRoot "README.md") $staging

# Bake the build arguments in: HF Spaces builds with no --build-arg flags.
$dockerfile = Join-Path $staging "Dockerfile"
(Get-Content $dockerfile) `
    -replace '^ARG INDEX_ROWS=""$', "ARG INDEX_ROWS=$IndexRows" `
    | Set-Content $dockerfile -Encoding utf8

Get-ChildItem -Recurse $staging -Include __pycache__, *.pyc -Force |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$bytes = (Get-ChildItem -Recurse -File $staging | Measure-Object -Sum Length).Sum
"staged {0:N1} MB in {1}" -f ($bytes / 1MB), $staging

# --- push ---------------------------------------------------------------------
Push-Location $staging
try {
    git init -q
    git checkout -q -b main
    git add -A
    git -c user.name=$User -c user.email="$User@users.noreply.huggingface.co" `
        commit -q -m "VoiceRAG API: voice-enabled RAG over ai4bharat/MSMARCO-XI"

    $remote = "https://${User}:${token}@huggingface.co/spaces/${User}/${Space}"
    git remote add origin $remote
    Write-Host "pushing to huggingface.co/spaces/$User/$Space ..." -ForegroundColor Cyan
    git push -q --force origin main
    Write-Host ""
    Write-Host "Pushed. Build logs: https://huggingface.co/spaces/$User/$Space?logs=build" -ForegroundColor Green
    Write-Host "Now set SARVAM_API_KEY / GROQ_API_KEY / OPENAI_API_KEY under" -ForegroundColor Yellow
    Write-Host "  https://huggingface.co/spaces/$User/$Space/settings" -ForegroundColor Yellow
}
finally {
    Pop-Location
    # The remote URL embeds the token; do not leave it on disk.
    Remove-Item -Recurse -Force (Join-Path $staging ".git") -ErrorAction SilentlyContinue
}
