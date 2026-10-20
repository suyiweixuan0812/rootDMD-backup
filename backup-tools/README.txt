GitHub target: suyiweixuan0812/rootDMD-backup (must be private)
Repository root: /rootDMD
Backup interval: 10 minutes after each run, using Supervisor.
GitHub deploy key authorized; automatic backup managed by Supervisor (rootdmd-git-backup).
Runs automatically while the server and Supervisor are running; stopped servers cannot back up.
Supervisor config: /rootDMD/backup-tools/supervisor.conf
Logs: /rootDMD/backup-tools/backup.log (rotated, excluded from Git).
Last successful push: /rootDMD/.git/backup-last-success
Manual backup: /rootDMD/backup-tools/backup.sh
Inspect history: git -C /rootDMD log --oneline -20
Pause backup before restoring: touch /rootDMD/.git/BACKUP_PAUSED
Restore selected code from a commit (first inspect git status and save pending work):
  git -C /rootDMD restore --source=COMMIT -- ReFormFlow-main
Then commit the restored state and resume:
  git -C /rootDMD add -A
  git -C /rootDMD -c user.name='DMD Backup' -c user.email='suyiweixuan0812@users.noreply.github.com' commit -m 'Restore code from COMMIT'
  rm /rootDMD/.git/BACKUP_PAUSED
No force push or automatic pull is used. Push failures return an error and keep local commits.
Git does not back up ignored model weights, virtual environments, caches or secrets.
