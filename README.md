# Why-Do-Single-Cell-Foundation-Models-Lose-to-the-Mean-A-Diagnostic-Study?
# Why Do Single-Cell Foundation Models Lose to the Mean?

Diagnostic study of scGPT and scFoundation on the Replogle K562
Perturb-seq dataset.

## Setup
pip install -r requirements.txt
pip install git+https://github.com/bowang-lab/scGPT.git

## Quick test (no downloads needed)
python smoke_test.py

## Full experiment
python -m experiments.run_all \
  --model-dir path/to/scgpt_weights \
  --output-dir results/full_run \
  --seed 42
