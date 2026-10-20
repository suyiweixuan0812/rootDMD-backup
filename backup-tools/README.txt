GitHub target: suyiweixuan0812/rootDMD-backup (must be private)
Repository root: /rootDMD
Planned backup interval: 10 minutes. Remote authentication and scheduler are not configured yet.
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
