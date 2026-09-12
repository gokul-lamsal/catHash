#!/usr/bin/env bash
set -euo pipefail

# The Makefile detects every installed NVIDIA GPU automatically. Override with:
#   NVCC_ARCH=89 ./build.sh
make -C "$(dirname "$0")" NVCC_ARCH="${NVCC_ARCH:-auto}"
