#!/bin/bash
# Install Rust compiler for tokenizers
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
# Upgrade pip to the latest version
pip install --upgrade pip
