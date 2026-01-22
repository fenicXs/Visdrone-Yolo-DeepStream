#!/usr/bin/env bash
set -euo pipefail

# Phase 2 (Jetson): run DeepStream app with your config.

deepstream-app -c configs/deepstream/deepstream_app_config.txt
