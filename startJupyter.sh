#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0

nvidia-smi
jupyter notebook /uCTGan --ip 0.0.0.0 --port 8888 --allow-root --no-browser