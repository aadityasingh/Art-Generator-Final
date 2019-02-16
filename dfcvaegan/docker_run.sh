#!/bin/bash

# Set up mounts & pass environment and commandline through
nvidia-docker run \
    -v /data:/data \
    -e CUDA_VISIBLE_DEVICES \
    aaditya \
    $*
