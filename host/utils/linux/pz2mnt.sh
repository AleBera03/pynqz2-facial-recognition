#!/bin/bash

MOUNT_POINT="/mnt/sshfs"
REMOTE_FOLDER="/home/root"

if [[ ! -d "$MOUNT_POINT" ]]; then
    sudo mkdir -p $MOUNT_POINT
fi
sudo chown $USER:$USER $MOUNT_POINT
sudo chmod 0775 $MOUNT_POINT

sshfs pz2:$REMOTE_FOLDER $MOUNT_POINT