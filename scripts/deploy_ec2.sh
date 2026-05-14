#!/usr/bin/env bash
# deploy_ec2.sh — Pull latest image and restart services on an EC2 instance.
#
# Usage (run ON the EC2 instance):
#   bash scripts/deploy_ec2.sh
#
# Prerequisites on the instance:
#   - Docker + Docker Compose plugin installed
#   - Git repo cloned to ~/RT-Market-Movement-Prediction
#   - ~/.env file (or .env in repo root) with API keys
#   - Logged in to ghcr.io:  echo "$TOKEN" | docker login ghcr.io -u <user> --password-stdin

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/RT-Market-Movement-Prediction}"
COMPOSE_FILE="${REPO_DIR}/docker-compose.yml"
API_HEALTH_URL="http://localhost:8000/health"
FRONTEND_HEALTH_URL="http://localhost:8501"
MAX_WAIT=60   # seconds to wait for API to become healthy

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── 1. Pull latest code ────────────────────────────────────────────────────── #
log "Pulling latest code from origin/main …"
cd "$REPO_DIR"
git fetch origin
git reset --hard origin/main
log "Now at commit $(git rev-parse --short HEAD)"

# ── 2. Pull new Docker images ──────────────────────────────────────────────── #
log "Pulling Docker images …"
docker compose -f "$COMPOSE_FILE" pull

# ── 3. Restart services ────────────────────────────────────────────────────── #
log "Stopping existing containers …"
docker compose -f "$COMPOSE_FILE" down --remove-orphans

log "Starting services in detached mode …"
docker compose -f "$COMPOSE_FILE" up -d

# ── 4. Wait for API health check ───────────────────────────────────────────── #
log "Waiting for API to become healthy (max ${MAX_WAIT}s) …"
elapsed=0
until curl -sf "$API_HEALTH_URL" > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        log "ERROR: API did not become healthy within ${MAX_WAIT}s"
        docker compose -f "$COMPOSE_FILE" logs --tail=50 app
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
log "API healthy at ${API_HEALTH_URL}"

# ── 5. Quick frontend reachability check ──────────────────────────────────── #
if curl -sf --max-time 5 "$FRONTEND_HEALTH_URL" > /dev/null 2>&1; then
    log "Frontend reachable at ${FRONTEND_HEALTH_URL}"
else
    log "WARNING: Frontend not yet reachable (may still be starting)"
fi

# ── 6. Show public IP and service URLs ─────────────────────────────────────── #
PUBLIC_IP=$(curl -sf --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
log "Deployment complete!"
log "  Public IP  : ${PUBLIC_IP}"
log "  API        : http://${PUBLIC_IP}:8000"
log "  API docs   : http://${PUBLIC_IP}:8000/docs"
log "  Frontend   : http://${PUBLIC_IP}:8501"
log "  MLflow UI  : http://${PUBLIC_IP}:5000"

# ── 7. Prune dangling images to reclaim disk space ─────────────────────────── #
log "Pruning dangling images …"
docker image prune -f
log "Done."
