#!/usr/bin/env bash

# Use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Exit immediately if any command fails
set -e

for test_budget in low medium high; do
    for seed in 0 1 2; do
        for phase in cubic hexagonal; do
            for n_particles in 216 64; do
                for budget in low medium high; do
                    echo "Running: seed=${seed}, phase=${phase}, n_particles=${n_particles}, budget=${budget}, test_budget=${test_budget}"
                    python test_mw.py \
                        --phase "${phase}" \
                        --n_particles "${n_particles}" \
                        --budget "${budget}" \
                        --test_budget "${test_budget}" \
                        --seed "${seed}"
                done
            done
        done
    done
done
