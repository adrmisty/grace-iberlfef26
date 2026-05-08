# GRACE & CasiMedicos-Arg (IberLEF 2026)

This repository contains the complete data pre-processing, inference, ensemble, and evaluation pipeline for **GRACE (Granular Recognition of Argumentative Clinical Evidence)** (Track 2) of IberLEF 2026. More info on the Shared Task at: [Codabench Competition](https://www.codabench.org/competitions/13280/)

The project is divided into two primary subsystems:

1. **CasiMedicos-Arg re-processing (`src/casimedicos/`)**: Handles the extension and alignment of relations from the original dataset, and normalized generation of multilingual permutations of the data.
2. **GRACE ICL Pipeline (`src/grace/`)**: Handles zero-shot and few-shot inference, global vs. per-task prompting, ensemble generation, post-processing to adhere to shared task submission requirements, and LLM evaluation across the three official GRACE subtasks.

---

## Project Structure

```text
GRACE-IBERLEF26/
├── data/
│   ├── casimedicos/         # CasiMedicos-Arg dataset
│   │   ├── raw/             # Monolingual baseline files (_ordered.jsonl)
│   │   └── splits/          # Mono/bi/multilingual splits generated (dev, test, train)
│   ├── grace/               # Official IberLEF track_1 and track_2 JSON files
│   └── unified/             # Unified dataset splits combining multiple sources
│   
├── model/                   # Generated outputs and predictions
│   ├── casimedicos/         # Model outputs on the CasiMedicos dataset
│   ├── grace/               # Model outputs on the GRACE dataset
│   └── unified/             # Model outputs on the Unified dataset
│       └── OpenAI/
│           └── gpt-4o/
│               ├── submission/  # Compiled JSON files ready for scoring
│               └── best_runs/   # S3 Ensembles crossing other models with GPT
│
├── src/
│   ├── casimedicos/         # Pre-processing & split generation
│   │   ├── main.py          
│   │   ├── splits.py        
│   │   └── relations.py     
│   ├── grace/               # ICL inference and evaluation
│   │   ├── eval/            # Custom metric calculation (metric.py)
│   │   ├── post/            # Output cleaning and submission compilation (submit.py)
│   │   ├── main.py          
│   │   ├── model.py         # LLM factory (OpenAI, Qwen, MedGemma, Gemini)
│   │   ├── prompts.py       # Global and Subtask-specific prompt definitions
│   │   ├── score.py         # Official IberLEF scoring script
│   │   └── task.py          # Subtask execution, global inference & ensemble logic
│   ├── case.py              # Parsers for both GRACE and CasiMedicos data schemas
│   └── config.py            
├── .gitignore
└── requirements.txt

```

---

## 🛠️ CasiMedicos-ARG Alignment & Split Generation

The CasiMedicos dataset consists of clinical cases annotated with BIO tags for argumentative components (Premises and Claims) and relation pairs. 

In its raw form, the dataset presented several structural inconsistencies that prevented immediate multilingual evaluation and training. Most critically, the argumentative relations (Support/Attack pairs) were *only* processed, aligned, and mapped for the English dataset. The translated splits contained the clinical text and the entity BIO tags, but completely lacked the explicit relation mappings connecting those entities. On the other hand, the English source data was provided in `.json` format, whereas the Spanish, French, and Italian translations were distributed as `.jsonl` files.

To solve this, the pre-processing pipeline dynamically normalizes these records, using the English gold standard as an anchor to map and project the aligned relations across all other languages. From there, it generates all possible bilingual and multilingual combinations for robust cross-lingual testing.

### Key Features
* **Cross-Lingual Relation Alignment**: Matches and maps the relation pairs from the English source to the Spanish, French, and Italian splits to ensure complete parity across the dataset.
* **Format Normalization**: Converts all identified nested records into the strict `{"id", "text", "labels"}` format for the unified `_ordered.jsonl` files.
* **Language Identification**: Ensures all case IDs and relation IDs end with their respective language code (e.g., `_en`, `_es`) to avoid ID collisions during unified inference.
* **Multilingual Combinations**: Automatically generates merged split files (e.g., `train_es-fr.jsonl`, `test_all.jsonl`).

### Usage

Run the data pipeline from the root directory:

```bash
python -m src.casimedicos.main --align --split
```

---

## 🚀 GRACE In-Context Learning

This module prompts Large Language Models (in zero and few-shot settings) to extract argumentative structures from clinical texts. It now supports both **Global Inference** (extracting S1, S2, and S3 in a single prompt) and **Split Inference** (task-by-task), compiling the outputs into the official IberLEF GRACE submission schema.

### Subtasks

* **Subtask 1 (Sentence Relevance)**: Binary classification determining if a sentence contains argumentative substance.
* **Subtask 2 (Argumentative Components)**: Exact span extraction for `Premises` and `Claims`.
* **Subtask 3 (Relation Classification)**: Classifying the relationship between identified entities as `Support` or `Attack`.

### Architecture & Dataset Routing

The pipeline dynamically supports distinct dataset structures via the `--dataset` toggle:

* `--dataset grace` (Default): Uses the official `track_2_*.json` files. Evaluates natively in Spanish.
* `--dataset casimedicos`: Uses the `_ordered.jsonl` and `_relations.jsonl` files. Dynamically reverse-engineers Subtask 1 ground truth from BIO tags.
* `--dataset unified`: Uses combined dataset splits for robust few-shot sampling and evaluation.

---

## Ensemble Pipeline

In argumentation extraction tasks, some models excel at entity span extraction (S2) but lack the logical reasoning for relations (S3), while LLMs like GPT-4o excel at reasoning over pairs.

The **Ensemble Pipeline** takes a compiled submission file from *any* model (e.g., Qwen, MedGemma), extracts its predicted Premises and Claims, generates every possible candidate pair, and feeds them into a powerful reasoning model (like OpenAI's GPT) using the native S3 prompt to predict the final relations.

### Usage Examples

**1. Standard Pipeline**
Run split or global subtasks [inference, post-processing, submission adn evaluation steps] for a specific model:

```bash
python -m src.grace.main --run --clean --submit --eval \
  --model OpenAI \
  --sizes gpt-4o-mini \
  --settings few_shot zero_shot \
  --tasks S1 S2 S3 global \
  --dataset unified \
  --n_examples 4

```

**2. Ensemble Pipeline**
Take the S1/S2 output from an existing file and uses another model to resolve the S3 relations:

```bash
python -m src.grace.main \
  --bestrun --other_predictions model/best_runs/grace-qwen3.54B-few-global.json \
  --model OpenAI \
  --sizes gpt-5.4-mini \
  --settings few_shot \
  --dataset unified \
  --n_examples 4
```

#### Pipeline Flags:

* `--run`: Executes ICL inference against the selected LLM.
* `--post`: Post-processes results, cleans model hallucination (e.g., `<think>` tags, markdown blocks) and normalizes outputs.
* `--submit`: Compiles the raw predictions into the official IberLEF submission JSON format. Handles both static claim injection and robust text-overlap matching for S3.
* `--eval`: Compares predictions against the ground truth and generates detailed metrics [*].
* `--bestrun`: Makes use of a different model's predictions to gather S3 argumentative entity relation extraction..
* `--tasks`: Space-separated list of tasks to run (`S1`, `S2`, `S3`, `global`).
* `--dataset`: Selects the underlying dataset schema (`grace`, `casimedicos`, `unified`) for example sampling in few-shot.

---

## 📊 Evaluation Metrics

The internal evaluator computes:

* **Subtask 1**: F1-score (Positive Class), Precision, Recall, Accuracy.
* **Subtask 2**: Exact Match F1 (*very punitive* metric tracking perfect character-level overlap).
* **Subtask 3**: Macro F1-score and Accuracy.

[*] **Official Scoring:** You can also make use of `src/grace/eval/score.py`. This is the official scoring function provided by the IberLEF task organizers and can be used to validate your compiled `submission.json` against the gold standard using their exact evaluation criteria.

---

## Author

**Adriana R. Flórez**
*Computational Linguist & Software Engineer*
[GitHub Profile](https://github.com/adrmisty) | [LinkedIn](https://linkedin.com/in/adriana-rodriguez-florez)

---

*Built with ❤️ using Python, the OpenAI/Gemini APIs and HuggingFace.*