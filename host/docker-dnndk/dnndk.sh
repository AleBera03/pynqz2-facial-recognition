#!/bin/bash

if [[ $1 == "--restart" || $1 == "-r" ]]; then
    docker restart dnndk-container
elif [[ -n $1 ]]; then
    echo "USAGE: dnndk.sh [--restart|-r]"
    exit 1
fi

docker exec -it dnndk-container bash