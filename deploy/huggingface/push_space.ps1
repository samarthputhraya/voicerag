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

# The frontend sources. The Dockerfile's first stage builds them, so without
# this the Space build fails at `COPY web/package.json` -- and if it were made
# optional instead, it would succeed and publish a bare JSON API.
#
# Only the sources: node_modules, .next and out are regenerated inside the
# image, and public/ is rebuilt from node_modules by `npm run vad-assets`.
# Copying node_modules here would push several hundred MB into a git repo.
$webSrc = Join-Path $repo "web"
$webDst = Join-Path $staging "web"
New-Item -ItemType Directory -Path $webDst | Out-Null
foreach ($item in @("app", "components", "lib", "scripts")) {
    Copy-Item -Recurse (Join-Path $webSrc $item) (Join-Path $webDst $item)
}
foreach ($f in @("package.json", "package-lock.json", "next.config.mjs",
                 "tsconfig.json", "next-env.d.ts")) {
    Copy-Item (Join-Path $webSrc $f) $webDst
}
if (-not (Test-Path (Join-Path $webDst "package-lock.json"))) {
    Write-Error "web/package-lock.json is missing; the image builds with 'npm ci'"
    exit 1
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

# `$ErrorActionPreference = "Stop"` governs CMDLET errors only. It does NOT stop
# the script when a NATIVE executable exits non-zero -- $PSNativeCommandUseError-
# ActionPreference is PowerShell 7.3+, and this runs under Windows PowerShell
# 5.1. Without an explicit check, a `git push` that fails (no Space yet, token
# missing write scope, HF rejecting it) fell straight through to the green
# "Pushed." banner and exited 0. The operator then opened the build-logs URL,
# found a 404, and debugged the wrong thing -- while the `finally` below had
# already deleted the commit.
function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments,
          [Parameter(Mandatory = $true)][string]$What)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (git exited $LASTEXITCODE)"
    }
}

Push-Location $staging
$pushed = $false
try {
    Invoke-Git @("init", "-q") "git init"
    Invoke-Git @("checkout", "-q", "-b", "main") "git checkout -b main"
    Invoke-Git @("add", "-A") "git add"
    Invoke-Git @("-c", "user.name=$User",
                 "-c", "user.email=$User@users.noreply.huggingface.co",
                 "commit", "-q", "-m",
                 "VoiceRAG: voice-enabled RAG over ai4bharat/MSMARCO-XI") "git commit"

    $remote = "https://${User}:${token}@huggingface.co/spaces/${User}/${Space}"
    Invoke-Git @("remote", "add", "origin", $remote) "git remote add"
    Write-Host "pushing to huggingface.co/spaces/$User/$Space ..." -ForegroundColor Cyan
    Invoke-Git @("push", "-q", "--force", "origin", "main") "git push"
    $pushed = $true

    Write-Host ""
    Write-Host "Pushed. Build logs: https://huggingface.co/spaces/$User/$Space?logs=build" -ForegroundColor Green
    Write-Host "Now set SARVAM_API_KEY / GROQ_API_KEY / OPENAI_API_KEY under" -ForegroundColor Yellow
    Write-Host "  https://huggingface.co/spaces/$User/$Space/settings" -ForegroundColor Yellow
}
catch {
    Write-Host ""
    Write-Host "PUSH FAILED: $_" -ForegroundColor Red
    Write-Host "Most likely causes, in order:" -ForegroundColor Yellow
    Write-Host "  1. The Space does not exist yet. Create it first at" -ForegroundColor Yellow
    Write-Host "     https://huggingface.co/new-space  (SDK: Docker)." -ForegroundColor Yellow
    Write-Host "     NOTE: creating a Docker Space requires a paid plan (PRO for a" -ForegroundColor Yellow
    Write-Host "     personal account). Static Spaces are the only free SDK." -ForegroundColor Yellow
    Write-Host "  2. The token lacks *write* scope." -ForegroundColor Yellow
    Write-Host "  3. The username or space name is wrong." -ForegroundColor Yellow
    Write-Host "The staged commit is kept at $staging so a retry costs nothing." -ForegroundColor Yellow
}
finally {
    Pop-Location
    # The remote URL embeds the token and must not be left on disk. On success
    # the whole repo goes; on failure only the URL does, so the commit survives
    # for a retry -- `git push` again after fixing the cause, rather than
    # re-running the script and re-entering the token.
    if ($pushed) {
        Remove-Item -Recurse -Force (Join-Path $staging ".git") -ErrorAction SilentlyContinue
    }
    elseif (Test-Path (Join-Path $staging ".git")) {
        Push-Location $staging
        & git remote set-url origin "https://huggingface.co/spaces/${User}/${Space}" 2>$null
        Pop-Location
    }
}
if (-not $pushed) { exit 1 }
