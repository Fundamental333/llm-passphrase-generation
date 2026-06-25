# LLM Passphrase Generation Pipeline

This repository contains a reproducible pipeline for generating and analyzing LLM-generated passphrase datasets.

## Model

`Qwen/Qwen2.5-1.5B-Instruct`

## Hardware Used So Far

- Google Colab
- Tesla T4 GPU

## Main Generation Parameters

| Parameter | Value |
|---|---:|
| TARGET_PER_FAMILY | 1250 |
| Number of prompt families | 40 |
| Raw candidates per chunk | 50000 |
| N_PER_PROMPT | 8 |
| BATCH_SIZE | 8 |
| MAX_NEW_TOKENS | 140 |
| Output format | `.jsonl.gz` |





## Repository Contents

```text
src/
  run_t4_llm_passphrase_v4.py
  analyze_clean_chunks.py

configs/
  default_generation_config.json

scripts/
  run_colab_author_example_10_15.sh
  run_colab_collaborator_100_105.sh

examples/
  sample_output_format.jsonl

outputs/
  .gitkeep
