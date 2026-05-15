#!/bin/bash
set -e
BACKUP="/volume1/docker/cat-water-dashboard-backup-20260507-164305"
LIVE="/volume1/docker/cat-water-dashboard"
echo "=== Rolling back cat-water-dashboard ==="
echo "From: $BACKUP"
cd "$LIVE"
echo "Stopping containers..."
docker compose down 2>/dev/null || true
echo "Restoring files..."
rsync -a --delete "$BACKUP"/ "$LIVE"/
echo "Starting containers..."
docker compose up -d
echo "=== Rollback complete ==="

