#!/bin/bash

echo "=== All nvcc in PATH ==="
which -a nvcc | xargs -I {} sh -c '{} --version'
echo

echo "=== System-wide CUDA installations ==="
for cuda in /usr/local/cuda*/bin/nvcc; do
    if [ -x "$cuda" ]; then
        echo "=== $cuda ==="
        $cuda --version
        echo
    fi
done

echo "=== User-level CUDA installations ==="
for cuda in $HOME/cuda*/bin/nvcc; do
    if [ -x "$cuda" ]; then
        echo "=== $cuda ==="
        $cuda --version
        echo
    fi
done

echo "=== CUDA Environment Variables ==="
env | grep -i cuda
echo

echo "=== Current active nvcc ==="
nvcc --version
echo

echo "=== NVIDIA Driver Info ==="
nvidia-smi | grep "CUDA Version" 