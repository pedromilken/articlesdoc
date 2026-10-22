# Evaluating LLM Models for Model-to-Text Transformation: Singletons, Ensembles, and Statistical Analysis

Code and analysis pipeline for the paper presented at **ENIAC 2026** (23rd National Meeting on Artificial and Computational Intelligence, co-located with BRACIS 2026, Cuiabá, Brazil). Paper ID 30800.

**Authors:** Pedro Henrique Bernardes Caetano Milken, Marcelo de Almeida Maia, Leandro Nogueira Couto, Nicole Maia Argondizzi, Marcelo Keese Albertini, Federal University of Uberlândia (UFU)

| | |
|---|---|
| E-print | arXiv (submission in progress, link coming soon) |
| Replication package | [doi:10.5281/zenodo.19979264](https://doi.org/10.5281/zenodo.19979264) |
| Proceedings | SBC OpenLib (SOL), coming soon |

## Overview

We evaluate eight LLMs (Claude Opus 4.7, Claude Sonnet 4.6, Claude Haiku 4.5, GPT-5.5, GPT-5.4, GPT-4, DeepSeek V4 Pro, DeepSeek V4 Flash) on the task of turning BPMN process models into natural-language descriptions (Model-to-Text, M2T). We compare single-shot generation with an automated self-refine cycle (generate → diagnose → rewrite) on all 45 test processes of the PET dataset (`patriziobellan/PETv11`). Temperature is a control factor. The design extends Klievtsova et al. (IEEE RE 2024).

**Main finding:** self-refine reduces task omission but increases misrepresentation and lowers semantic fidelity.

## Requirements

Python 3.10+ and:

```
pip install numpy pandas requests scipy scikit-learn statsmodels tenacity tqdm \
            python-dotenv sacrebleu sacremoses sentence-transformers datasets matplotlib
```

API keys are read from environment variables (or a local `.env` file, which must never be committed):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Usage

```
python m2t_petv11_unico.py validate-models                 # check model IDs
python m2t_petv11_unico.py experiment --sample-size 45      # run the factorial experiment
python m2t_petv11_unico.py analysis --input outputs/results.csv
python m2t_petv11_unico.py plots    --input outputs/results.csv
```

Other subcommands: `list-models`, `inspect-dataset`.

## Metrics

| KPI | Meaning |
|---|---|
| TS-C | Contextual similarity (multilingual sentence embeddings) between the generated text and the original PET description |
| TS-NC | Non-contextual similarity (TF-IDF) |
| BLEU-1 | Lexical overlap |
| O-OM | Omission: expected tasks with similarity < 0.18 |
| O-OO | Misrepresentation: expected tasks with 0.18 ≤ similarity < 0.40 |
| O-OT | Ordering error (normalised Kendall-tau distance) |
| O-OG | Spurious generation: sentences matching no expected task |
| MR | Modeling-register leakage (binary) |

## Citation

```bibtex
@inproceedings{milken2026m2t,
  title     = {Evaluating {LLM} Models for Model-to-Text Transformation: Singletons, Ensembles, and Statistical Analysis},
  author    = {Milken, Pedro Henrique Bernardes Caetano and Maia, Marcelo de Almeida and Couto, Leandro Nogueira and Argondizzi, Nicole Maia and Albertini, Marcelo Keese},
  booktitle = {Anais do XXIII Encontro Nacional de Intelig{\^e}ncia Artificial e Computacional (ENIAC)},
  year      = {2026},
  address   = {Cuiab{\'a}, MT, Brazil},
  publisher = {Sociedade Brasileira de Computa{\c{c}}{\~a}o}
}
```

## Funding

CAPES (PAPG/PPGCO-UFU, fellowship 15465) and CNPq (Grant 306795/2022).
