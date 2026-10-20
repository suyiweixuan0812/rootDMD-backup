#!/usr/bin/env bash
set -euo pipefail
cd /rootDMD
exec 9>/tmp/rootDMD-git-backup.lock
flock -n 9 || exit 0
if [[ -e .git/BACKUP_PAUSED ]]; then exit 0; fi
if [[ -n "$(git ls-files -u)" ]]; then
  echo 'Unresolved merge conflicts; backup stopped.' >&2
  exit 1
fi
git add -A
if ! git diff --cached --quiet; then
  git -c user.name='DMD Backup' -c user.email='suyiweixuan0812@users.noreply.github.com' commit -m "Automatic backup $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
git push origin main
