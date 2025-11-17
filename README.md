PerformanceDNA with GNNs

How to copy your entire project from remote SHH machine to your local machine.

1. Open a terminal on local machine.
2. Make a `.rsyncignore` file
> venv/  
  __pycache__/  
  *.pt  
  *.log  
  .git/
2. Run `rsync` to copy/update the whole folder on local terminal.

> rsync -avz --progress \
  --exclude-from='.rsyncignore' \
  your_username@remote.server:/path/to/remote/project/ \
  /path/to/local/project/

3. Commit and push.

You can make a 