#!/bin/bash

# Sync the remote directory into the current local directory.

REMOTE="synergy3_new_difei"
REMOTE_DIR="/usr/scratch/dcao48/difei/sushanth/gnns/profiling/"
LOCAL_DIR="."

echo "Starting rsync from $REMOTE:$REMOTE_DIR to $LOCAL_DIR"
echo "Using exclude file: .rsyncignore"
echo

rsync -avz --progress \
    --exclude-from='.rsyncignore' \
    "$REMOTE:$REMOTE_DIR" \
    "$LOCAL_DIR"

echo
echo "Sync complete."