#!/usr/bin/env python3
"""Run all chronological ML experiments: python run_ml.py."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
from ml_experiment.run import main

if __name__ == '__main__':
    main(ROOT)
