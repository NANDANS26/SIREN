#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

# Colour support
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { printf "%b\n" "$*"; }
pass() { printf "${GREEN}✓ %b${NC}\n" "$*"; }
fail() { printf "${RED}✗ %b${NC}\n" "$*"; }
hint() { printf "${YELLOW}  hint: %b${NC}\n" "$*"; }

FAILED_AT=""
stop_at() { FAILED_AT="$1"; exit 1; }

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

# ── args ─────────────────────────────────────────────────────────────────────

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

if [ -z "$PING_URL" ]; then
  log "${BOLD}Usage:${NC} $0 <ping_url> [repo_dir]"
  log "  ping_url  Your HuggingFace Space URL (e.g. https://your-space.hf.space)"
  log "  repo_dir  Path to your repo (default: current directory)"
  exit 1
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"
log "Repo:     $REPO_DIR"
log "Space:    $PING_URL"
printf "\n"

# ── Step 1: Ping HF Space ────────────────────────────────────────────────────

log "${BOLD}Step 1/3: Pinging HF Space${NC} ..."

PING_OK=false
for attempt in 1 2 3; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 30 \
    "${PING_URL%/}/health" 2>/dev/null) && true
  if [ "$HTTP_CODE" = "200" ]; then
    PING_OK=true
    break
  fi
  log "  Attempt $attempt: got HTTP $HTTP_CODE, retrying in 5s..."
  sleep 5
done

if [ "$PING_OK" = true ]; then
  pass "HF Space is live (HTTP 200 on /health)"
else
  fail "HF Space did not respond with HTTP 200 on ${PING_URL%/}/health (got: $HTTP_CODE)"
  hint "Make sure your Space is running and the URL is correct."
  hint "The server must expose GET /health returning {\"status\": \"ok\"}"
  stop_at "Step 1"
fi

# ── Step 2: Docker build ─────────────────────────────────────────────────────

log "${BOLD}Step 2/3: Building Docker image${NC} ..."

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

# ── Step 3: openenv validate ─────────────────────────────────────────────────

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
