# eval.py
# ----------------------------------------------------------------------------------------
# evaluation metrics calculation for few-shot/zero-shot testing on GRACE/CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import re
import json
import logging
import config as settings
from pathlib import Path
from typing import List, Dict, Any
from case import load_cases, load_relations

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

class GraceEvaluator:
    def __init__(self, splits_dir: str = settings.SPLITS_DATA_DIR):
        self.splits_dir = Path(splits_dir)

    # --- SUBTASK 1 -------------------------------------------------------------------------

    def evaluate_subtask_1(self, predictions_path: Path, ground_truth_path: Path):
        """SUBTASK 1: Sentence relevance detection (F1-score Positive Class)."""
        logging.info(f"> Evaluating Subtask 1 metrics for: {predictions_path.name}")

        gt_map = {case["id"]: case["relevance_labels"] for case in load_cases(ground_truth_path)}
        raw_predictions = self._load_json(predictions_path)

        tp, fp, tn, fn = 0, 0, 0, 0

        for record in raw_predictions:
            case_id = record["id"]
            if case_id not in gt_map:
                continue

            pred_labels = self._normalize_s1_prediction(record["prediction"])
            true_labels = gt_map[case_id]

            for sent_idx, true_val in true_labels.items():
                pred_val = pred_labels.get(sent_idx, False)

                if true_val and pred_val: tp += 1
                elif not true_val and not pred_val: tn += 1
                elif not true_val and pred_val: fp += 1
                elif true_val and not pred_val: fn += 1

        self._log_metrics("SUBTASK 1", tp, fp, fn, tn, f1_name="F1-score (Positive Class)")

    # --- SUBTASK 2 -------------------------------------------------------------------------
    
    def evaluate_subtask_2(self, predictions_path: Path, ground_truth_path: Path):
        """SUBTASK 2: Span detection (Exact Match F1)."""
        logging.info(f"> Evaluating Subtask 2 metrics for: {predictions_path.name}")

        gt_map = {c["id"]: c for c in load_cases(ground_truth_path)}
        preds = self._load_json(predictions_path)

        tp, fp, fn = 0, 0, 0

        for record in preds:
            case_id = record["id"]
            if case_id not in gt_map:
                continue

            pred_spans = set(self._extract_predicted_spans(record["prediction"]))
            
            gt_spans_raw = gt_map[case_id].get("premises", []) + gt_map[case_id].get("claims", [])
            gt_spans = set(self._normalize_span(s) for s in gt_spans_raw)

            tp += len(pred_spans & gt_spans)
            fp += len(pred_spans - gt_spans)
            fn += len(gt_spans - pred_spans)

        self._log_metrics("SUBTASK 2", tp, fp, fn, f1_name="Exact Match F1")

    # --- SUBTASK 3 -------------------------------------------------------------------------

    def evaluate_subtask_3(self, predictions_path: Path, ground_truth_path: Path):
        """SUBTASK 3: Relation detection (Macro F1-score)."""
        logging.info(f"> Evaluating Subtask 3 metrics for: {predictions_path.name}")

        preds = self._load_json(predictions_path)
        gt_map = {x["id"]: x["label"] for x in load_relations(ground_truth_path)}

        y_true, y_pred = [], []

        for record in preds:
            rid = record["id"]
            if rid not in gt_map:
                continue

            y_pred.append(self._normalize_relation(record["prediction"]))
            y_true.append(self._normalize_relation(gt_map[rid]))

        classes = ["Support", "Attack"]
        f1_scores = []

        for c in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1_scores.append(2 * p * r / (p + r) if (p + r) else 0.0)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0

        logging.info(f"\n\t --- SUBTASK 3 RESULTS ---")
        logging.info(f"\t Total relations: {len(y_true)}")
        logging.info(f"\t -------------------------")
        logging.info(f"\t Accuracy:       {accuracy:.4f}")
        logging.info(f"\t Macro F1-score: {macro_f1:.4f}\n")

    # --- log results -----------------------------------------------------------------------

    def _log_metrics(self, task_name: str, tp: int, fp: int, fn: int, tn: int = None, f1_name: str = "F1-score"):
        """Calculates and logs evaluation metrics dynamically based on the requested metric name."""
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        
        logging.info(f"\n\t --- {task_name} RESULTS ---")
        if tn is not None:
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
            logging.info(f"\t Total items evaluated: {tp + tn + fp + fn}")
            logging.info(f"\t TP: {tp}")
            logging.info(f"\t FP: {fp}")
            logging.info(f"\t TN: {tn}")
            logging.info(f"\t FN: {fn}")
            logging.info(f"\t -------------------------")
            logging.info(f"\t Precision: {precision:.4f}")
            logging.info(f"\t Recall:    {recall:.4f}")
            logging.info(f"\t {f1_name}: {f1:.4f}")
            logging.info(f"\t Accuracy:  {accuracy:.4f}\n")
        else:
            logging.info(f"\t TP: {tp}")
            logging.info(f"\t FP: {fp}")
            logging.info(f"\t FN: {fn}")
            logging.info(f"\t -------------------------")
            logging.info(f"\t Precision: {precision:.4f}")
            logging.info(f"\t Recall:    {recall:.4f}")
            logging.info(f"\t {f1_name}: {f1:.4f}\n")

    # --- io & normalization ----------------------------------------------------------------

    def _load_json(self, path: Path) -> Any:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_think_tags(self, text: str) -> str:
        """Removes the reasoning process hallucinations generated by reasoning models."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.replace("</think>", "")

    def _normalize_s1_prediction(self, prediction: str) -> Dict[str, bool]:
        if isinstance(prediction, str):
            prediction = self._clean_think_tags(prediction).strip()
            match = re.search(r"\{.*\}", prediction, re.DOTALL)
            if match:
                prediction = match.group()
            try:
                prediction = json.loads(prediction)
            except json.JSONDecodeError:
                logging.warning("\t> (!) Failed to parse prediction JSON.")
                return {}

        normalized = {}
        for key, value in prediction.items():
            sent_id = re.sub(r"[\[\]]", "", str(key))
            if isinstance(value, str):
                value = value.lower() == "true"
            normalized[sent_id] = bool(value)

        return normalized

    def _extract_predicted_spans(self, prediction: str) -> List[str]:
        prediction = self._clean_think_tags(prediction)
        spans = re.findall(r"\[(.*?)\]", prediction, re.DOTALL)
        if not spans:
            spans = re.findall(r"-\s*(.*)", prediction)
        return [self._normalize_span(s) for s in spans if s.strip()]

    def _normalize_span(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\wáéíóúñü ]", "", text)
        return text

    def _normalize_relation(self, label: str, pos_verb: str = "Support", neg_verb: str = "Attack") -> str:
        label = self._clean_think_tags(label).lower().strip()
        if "support" in label: return pos_verb
        if "attack" in label: return neg_verb
        return ""