#!/usr/bin/env bash

# Use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Exit immediately if any command fails
set -e

for seed in 0 1 2; do
    for phase in hcp; do
        for n_particles in 180; do
            for budget in high medium low; do
                echo "Running: seed=${seed}, phase=${phase}, n_particles=${n_particles}, budget=${budget}"
                python train_lj.py \
                    --phase "${phase}" \
                    --n_particles "${n_particles}" \
                    --budget "${budget}" \
                    --seed "${seed}"
            done
        done
    done
done
