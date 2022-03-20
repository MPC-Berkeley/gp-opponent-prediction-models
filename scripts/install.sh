#!/bin/bash
PATH=$PATH:~/.local/bin
echo "PATH=$PATH:~/.local/bin" >> ~/.bashrc

python3 -m pip install -e .