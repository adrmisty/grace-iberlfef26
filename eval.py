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
from collections import Counter
from case import load_cases, load_relations

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

class GraceEvaluator:
    def __init__(self, splits_dir: str = settings.SPLITS_DATA_DIR):
        self.splits_dir = Path(splits_dir)

    # --- SUBTASK 1 -------------------------------------------------------------------------

    def evaluate_subtask_1(self, predictions_path: Path, ground_truth_path: Path):
        """Precision, recall, and F1 for SUBTASK 1: sentence relevance detection."""
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

        self._log_metrics("SUBTASK 1", tp, fp, fn, tn)

    # --- SUBTASK 2 -------------------------------------------------------------------------
    
    def evaluate_subtask_2(self, predictions_path: Path, ground_truth_path: Path):
        """Accuracy for subtask 2 predictions (argument entity classification: PREMISE/CLAIM) accuracy."""
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

        self._log_metrics("SUBTASK 2", tp, fp, fn)

    # --- SUBTASK 3 -------------------------------------------------------------------------

    def evaluate_subtask_3(self, predictions_path: Path, ground_truth_path: Path):
        """Accuracy for subtask 3 predictions (relation detection: SUPPORT/ATTACK) accuracy."""
        logging.info(f"> Evaluating Subtask 3 metrics for: {predictions_path.name}")

        preds = self._load_json(predictions_path)
        gt_map = {x["id"]: x["label"] for x in load_relations(ground_truth_path)}

        correct, incorrect = 0, 0

        for record in preds:
            rid = record["id"]
            if rid not in gt_map:
                continue

            pred = self._normalize_relation(record["prediction"])
            true = self._normalize_relation(gt_map[rid])

            if true == pred: correct += 1
            else: incorrect += 1

        total = correct + incorrect
        accuracy = correct / total if total else 0.0

        logging.info(f"\n\t --- SUBTASK 3 RESULTS ---")
        logging.info(f"\t Total relations: {total}")
        logging.info(f"\t Correct: {correct}")
        logging.info(f"\t Incorrect: {incorrect}")
        logging.info(f"\t -------------------------")
        logging.info(f"\t Accuracy: {accuracy:.4f}\n")

    # --- log results -------------------------------------------------------------------------

    def _log_metrics(self, task_name: str, tp: int, fp: int, fn: int, tn: int = None):
        """Calculates and logs evaluation metrics to avoid redundancy."""
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
            logging.info(f"\t F1-score:  {f1:.4f}")
            logging.info(f"\t Accuracy:  {accuracy:.4f}\n")
        else:
            logging.info(f"\t TP: {tp}")
            logging.info(f"\t FP: {fp}")
            logging.info(f"\t FN: {fn}")
            logging.info(f"\t -------------------------")
            logging.info(f"\t Precision: {precision:.4f}")
            logging.info(f"\t Recall:    {recall:.4f}")
            logging.info(f"\t F1-score:  {f1:.4f}\n")

    # --- io & normalization -------------------------------------------------------------------------

    def _load_json(self, path: Path) -> Any:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _normalize_s1_prediction(self, prediction: str) -> Dict[str, bool]:
        """Normalizes the raw prediction results into a dictionary of sentence relevance labels."""
        if isinstance(prediction, str):
            prediction = prediction.strip()
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
            normalized[sent_id] = bool(value)

        return normalized

    def _extract_predicted_spans(self, prediction: str) -> List[str]:
        """Extracts predicted premise/claim spans."""
        spans = re.findall(r"\[(.*?)\]", prediction, re.DOTALL)
        if not spans:
            spans = re.findall(r"-\s*(.*)", prediction)
        return [self._normalize_span(s) for s in spans if s.strip()]

    def _normalize_span(self, text: str) -> str:
        """Normalizes extracted spans into lowercase sentences."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\wáéíóúñü ]", "", text)
        return text

    def _normalize_relation(self, label: str, pos_verb: str = "Support", neg_verb: str = "Attack") -> str:
        """Normalizes relation labels to Support, Attack, or nothing."""
        label = label.lower().strip()
        if "support" in label: return pos_verb
        if "attack" in label: return neg_verb
        return ""