#!/bin/bash

# Set up mounts & pass environment and commandline through
nvidia-docker run \
    -v $HOME/Art-Generator-Final/dfcvaegan:/dfcvaegan \
    -e CUDA_VISIBLE_DEVICES \
    aaditya \
    $*
