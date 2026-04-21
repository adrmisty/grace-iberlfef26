# GRACE & CasiMedicos-Arg (IberLEF 2026)

This repository contains the complete data pre-processing, inference, and evaluation pipeline for **GRACE (Granular Recognition of Argumentative Clinical Evidence)** (Track 2) of IberLEF 2026. More info on the Shared Task at: https://www.codabench.org/competitions/13280/

The project is divided into two primary subsystems:
1. **CasiMedicos-Arg re-processing (`src/casimedicos/`)**: Handles the extension and alignment of relations from the original dataset, and normalized generation of multilingual permutations of the data.
2. **GRACE ICL Pipeline (`src/grace/`)**: Handles zero-shot and few-shot inference, post-processing of results to adhere to shared task submission requirements, and evaluation of LLMs across the three official GRACE subtasks of the chosen track.

---

## Project Structure

```text
GRACE-IBERLEF26/
├── data/
│   ├── casimedicos/         # CasiMedicos-Arg dataset
│   │   ├── raw/             # Monolingual baseline files (_ordered.jsonl)
│   │   └── splits/          # Mono/bi/multilingual splits generated (dev, test, train)
│   ├── grace/               # *Official IberLEF track_1 and track_2 JSON files
│   
|
├── model/                   # Generated outputs and predictions
│   ├── casimedicos/         # Model outputs on the CasiMedicos dataset
│   └── grace/               # Model outputs on the GRACE dataset
│       ├── en/               
│       └── es/              
├── src/
│   ├── casimedicos/         # Pre-processing & split generation
│   │   ├── main.py          
│   │   ├── splits.py        
│   │   └── relations.py     
│   ├── grace/               # ICL inference and evaluation
│   │   ├── eval.py          # Custom metric calculation
│   │   ├── main.py          
│   │   ├── model.py         # LLM factory (OpenAI, Qwen, MedGemma, Gemini)
│   │   ├── post.py          # Submission compilation
│   │   ├── prompts.py       
│   │   ├── score.py         # *Official IberLEF scoring script
│   │   └── task.py          # Subtask execution & dataset routing
│   ├── case.py              # Parsers for both GRACE and CasiMedicos data schemas
│   └── config.py           
├── .gitignore
└── requirements.txt
```

---

## 🛠️ [1] CasiMedicos-ARG Alignment & Split Generation

The CasiMedicos dataset consists of clinical cases annotated with BIO tags for argumentative components (Premises and Claims) and relation pairs. The pre-processing pipeline dynamically normalizes these records in order to achieve aligned relations for all languages and generates all possible bilingual and multilingual combinations.

### Key Features
* **Relation Alignment**: Maps aligned relations for all languages with English as a base, as the original relations records only contained the English data.
* **Format Normalization**: Converts all identified nested records them into the strict `{"id", "text", "labels"}` format for the `_ordered.jsonl` files.
* **Language Identification**: Ensures all case IDs and relation IDs end with their respective language code (e.g., `_en`, `_es`).
* **Multilingual Combinations**: Automatically generates merged split files (e.g., `train_es-fr.jsonl`, `test_all.jsonl`).

### Usage

Run the data pipeline from the root directory:

```bash
python -m src.casimedicos.main --align --split
```

---

## 🚀 [2] GRACE In-Context Learning

This module prompts Large Language Models (in zero and few-shot settings) to extract argumentative structures from clinical texts and compiles the outputs into the official IberLEF GRACE submission schema for Track 2.

### Subtasks
* **Subtask 1 (Sentence Relevance)**: Binary classification determining if a sentence contains argumentative substance for deriving clinical treatment or diagnoses.
* **Subtask 2 (Argumentative Components)**: Exact span extraction for `Premises` and `Claims`.
* **Subtask 3 (Relation Classification)**: Classifying the relationship between identified entities as `Support` or `Attack`.

### Architecture & Dataset Routing
The pipeline dynamically supports distinct dataset structures via the `--dataset` toggle:
* `--dataset grace` (Default): Uses the official `track_2_*.json` files. Evaluates natively in Spanish.
* `--dataset casimedicos`: Uses the `_ordered.jsonl` and `_relations.jsonl` files. Dynamically reverse-engineers Subtask 1 ground truth from BIO tags and builds a GRACE-compatible submission file.

### Usage

Run the full end-to-end evaluation:

```bash
python -m src.grace.main --run --post --submit --eval \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot \
  --dataset casimedicos
```

#### Pipeline Flags:
* `--run`: Executes ICL inference against the selected LLM.
* `--post`: Cleans model hallucination (e.g., `<think>` tags) and normalizes JSON outputs.
* `--submit`: Compiles the separate S1, S2, and S3 outputs into a single, unified submission file. 
* `--eval`: Compares predictions against the ground truth and generates detailed metrics (`eval.txt`).

---

## Evaluation Metrics

The internal evaluator computes:
* **Subtask 1**: F1-score (Positive Class), Precision, Recall, Accuracy.
* **Subtask 2**: Exact Match F1 (*very punitive* metric tracking perfect character-level overlap).
* **Subtask 3**: Macro F1-score and Accuracy.

All evaluation logs are appended automatically to `eval.txt` in the respective model's output directory.

**Official Scoring:** You can also make use of `src/grace/score.py`. This is the official scoring function provided by the IberLEF task organizers and can be used to validate your compiled `submission.json` against the gold standard using their exact evaluation criteria.

## Author

**Adriana R. Flórez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

---

*Built with ❤️ using Python, the OpenAI/Gemini APIs and HuggingFace.*
