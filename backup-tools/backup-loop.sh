#!/usr/bin/env bash
set -uo pipefail
exec 8>/tmp/rootDMD-git-backup-loop.lock
flock -n 8 || exit 0
while true; do
  printf '\n[%s] Starting backup\n' "$(date -u +%FT%TZ)"
  if [[ -e /rootDMD/.git/BACKUP_PAUSED ]]; then
    echo "Backup paused."
    sleep 600 &
    wait $!
    continue
  fi
  if timeout --signal=TERM --kill-after=30s 300 /rootDMD/backup-tools/backup.sh; then
    date -u +%FT%TZ > /rootDMD/.git/backup-last-success
    echo 'Backup completed.'
  else
    result=$?
    printf 'Backup failed (exit %s); local commits kept; retry in 600 seconds.\n' "$result"
  fi
  sleep 600 &
  wait $!
done
