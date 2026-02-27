# Hybrid Peptide Structure Prediction Pipeline

This repository contains an automated, hybrid computational pipeline for predicting the 3D structures of chemically modified peptides, specifically those containing non-canonical amino acids (NCAAs) and other synthetic modifications.

## 🧬 Pipeline Architecture

This tool automates a 5-step process to generate highly accurate structural predictions:
1. **Sequence Parsing:** Maps canonical backbones and identifies non-canonical modifications.
2. **Backbone Prediction:** Predicts the canonical backbone structure.
3. **Side-Chain Generation (ETFlow):** Generates 3D conformers for non-canonical amino acids using the ETFlow Chemical Language Model.
4. **Stitching (BioPython):** Mathematically aligns and snaps the modified side-chains onto the predicted backbone.
5. **Minimization (OpenBabel):** Relaxes the final structure to resolve spatial clashes and optimize physical geometry.

## 🚀 Quick Start (Google Colab)

The easiest way to run this pipeline without local installation is via our pre-configured Google Colab environment.

1. Open the `run_peptide_pipeline.ipynb` notebook in Google Colab.
2. Run the environment setup cells to automatically build Conda, ETFlow, and necessary dependencies.
3. Execute the master script:
   ```bash
   python main.py --sequence "KET(AIB)AAKFERQHLDS" --json "modifications.json"
