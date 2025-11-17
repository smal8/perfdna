PerformanceDNA with GNNs

How to copy your entire project from remote SHH machine to your local machine.

1. Open a terminal on local machine.
2. Run `rsync` to copy/update the whole folder on local terminal.

> rsync -avz --progress \
  your_username@remote.server:/path/to/remote/project/ \
  /path/to/local/project/

3. Commit and push