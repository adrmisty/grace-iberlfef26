# eval.py
# ----------------------------------------------------------------------------------------
# evaluation metrics calculation for few-shot/zero-shot testing on GRACE/CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.case import load_cases, load_relations, load_cases_casiMedicos, load_relations_casiMedicos

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
EVAL_PATH = "eval.txt"

class GraceEvaluator:
    def __init__(self):
        pass

    # --- SUBTASK 1 -------------------------------------------------------------------------

    def evaluate_subtask_1(self, predictions_path: Path, ground_truth_path: Path, dataset: str = "grace"):
        """SUBTASK 1: Sentence relevance detection (F1-score positive class)."""
        header = f"> Evaluating Subtask 1 metrics for: {predictions_path.name}"
        logging.info(header)
        
        out_file = predictions_path.parent / EVAL_PATH
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"\n{header}\n")

        # Dynamically select loader
        if dataset == "casimedicos":
            cases = load_cases_casiMedicos(ground_truth_path)
        else:
            cases = load_cases(ground_truth_path)

        gt_map = {case["id"]: case["relevance_labels"] for case in cases}
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

        self._log_metrics("SUBTASK 1", tp, fp, fn, tn, f1_name="F1-score (Positive class)", out_file=out_file)

    # --- SUBTASK 2 -------------------------------------------------------------------------
    
    def evaluate_subtask_2(self, predictions_path: Path, ground_truth_path: Path, dataset: str = "grace"):
        """SUBTASK 2: Span detection (exact match F1)."""
        header = f"> Evaluating Subtask 2 metrics for: {predictions_path.name}"
        logging.info(header)
        
        out_file = predictions_path.parent / EVAL_PATH
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"\n{header}\n")

        # Dynamically select loader
        if dataset == "casimedicos":
            cases = load_cases_casiMedicos(ground_truth_path)
        else:
            cases = load_cases(ground_truth_path)

        gt_map = {c["id"]: c for c in cases}
        preds = self._load_json(predictions_path)

        tp, fp, fn = 0, 0, 0

        for record in preds:
            case_id = record["id"]
            if case_id not in gt_map:
                continue

            pred_spans = set(self._extract_predicted_spans(record["prediction"]))
            
            # Extract ground truth spans, safely handling dict vs string lists
            gt_spans_raw = gt_map[case_id].get("premises", []) + gt_map[case_id].get("claims", [])
            gt_spans_clean = [s.get("text", "") if isinstance(s, dict) else s for s in gt_spans_raw]
            gt_spans = set(self._normalize_span(s) for s in gt_spans_clean if s)

            tp += len(pred_spans & gt_spans)
            fp += len(pred_spans - gt_spans)
            fn += len(gt_spans - pred_spans)

        self._log_metrics("SUBTASK 2", tp, fp, fn, f1_name="Exact Match F1", out_file=out_file)

    # --- SUBTASK 3 -------------------------------------------------------------------------

    def evaluate_subtask_3(self, predictions_path: Path, ground_truth_path: Path, dataset: str = "grace"):
        """SUBTASK 3: Relation detection (Macro F1-score)."""
        header = f"> Evaluating Subtask 3 metrics for: {predictions_path.name}"
        logging.info(header)
        
        out_file = predictions_path.parent / EVAL_PATH
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"\n{header}\n")

        # Dynamically derive relations file if using CasiMedicos
        if dataset == "casimedicos":
            rel_name = ground_truth_path.stem.replace("_ordered", "_relations") + ".jsonl"
            rel_path = ground_truth_path.with_name(rel_name)
            relations = load_relations_casiMedicos(rel_path)
        else:
            relations = load_relations(ground_truth_path)

        preds = self._load_json(predictions_path)
        gt_map = {x["id"]: x["label"] for x in relations}

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

        lines = [
            f"\n\t --- SUBTASK 3 RESULTS ---",
            f"\t Total relations: {len(y_true)}",
            f"\t -------------------------",
            f"\t Accuracy:       {accuracy:.4f}",
            f"\t Macro F1-score: {macro_f1:.4f}\n"
        ]
        
        output_str = "\n".join(lines)
        logging.info(output_str)
        
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(output_str + "\n")

    # --- log results -----------------------------------------------------------------------

    def _log_metrics(self, task_name: str, tp: int, fp: int, fn: int, tn: int = None, f1_name: str = "F1-score", out_file: Path = None):
        """Calculates and logs evaluation metrics dynamically based on the requested metric name."""
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        
        lines = [f"\n\t --- {task_name} RESULTS ---"]
        
        if tn is not None:
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
            lines.extend([
                f"\t Total items evaluated: {tp + tn + fp + fn}",
                f"\t TP: {tp}",
                f"\t FP: {fp}",
                f"\t TN: {tn}",
                f"\t FN: {fn}",
                f"\t -------------------------",
                f"\t Precision: {precision:.4f}",
                f"\t Recall:    {recall:.4f}",
                f"\t {f1_name}: {f1:.4f}",
                f"\t Accuracy:  {accuracy:.4f}\n"
            ])
        else:
            lines.extend([
                f"\t TP: {tp}",
                f"\t FP: {fp}",
                f"\t FN: {fn}",
                f"\t -------------------------",
                f"\t Precision: {precision:.4f}",
                f"\t Recall:    {recall:.4f}",
                f"\t {f1_name}: {f1:.4f}\n"
            ])

        output_str = "\n".join(lines)
        logging.info(output_str)
        
        if out_file:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(output_str + "\n")

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
                return {}

        normalized = {}
        for key, value in prediction.items():
            sent_id = re.sub(r"[\[\]]", "", str(key))
            if isinstance(value, str):
                value = value.lower() == "true"
            normalized[sent_id] = bool(value)

        return normalized

    def _extract_predicted_spans(self, prediction: Any) -> List[str]:
        if isinstance(prediction, dict):
            raw_spans = prediction.get("premises", []) + prediction.get("claims", []) + \
                        prediction.get("Premisas", []) + prediction.get("Afirmaciones", []) + prediction.get("Claims", [])
            
            # Safely unpack dictionaries (like claims) vs strings (like premises)
            clean_spans = []
            for s in raw_spans:
                if isinstance(s, dict):
                    clean_spans.append(s.get("text", ""))
                elif isinstance(s, str):
                    clean_spans.append(s)
            
            return [self._normalize_span(s) for s in clean_spans if s.strip()]
            
        prediction = self._clean_think_tags(str(prediction))
        spans = re.findall(r"\[(.*?)\]", prediction, re.DOTALL)
        if not spans:
            spans = re.findall(r"-\s*(.*)", prediction)
        return [self._normalize_span(s) for s in spans if s.strip()]

    def _normalize_relation(self, label: Any, pos_verb: str = "Support", neg_verb: str = "Attack") -> str:
        if isinstance(label, dict):
            label = label.get("label", "")
            
        label = self._clean_think_tags(str(label)).lower().strip()
        if "support" in label: return pos_verb
        if "attack" in label: return neg_verb
        return ""
    
    def _normalize_span(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\wáéíóúñü ]", "", text)
        return text