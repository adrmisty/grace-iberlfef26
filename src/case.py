# case.py
# ----------------------------------------------------------------------------------------
# clinical case and relations parsing for https://huggingface.co/datasets/HiTZ/casimedicos-arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def load_cases(file_path: Path) -> List[Dict[str, Any]]:
    """Loads and parses BIO-tagged clinical cases."""
    logging.info(f"> Loading cases from {file_path.name}")
    parsed_cases = []

    if not file_path.exists():
        logging.error(f"\t (!) File not found: {file_path}")
        return parsed_cases

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw_record = json.loads(line)

            if "id" in raw_record or "case_id" in raw_record:
                case_id = str(raw_record.get("id", raw_record.get("case_id")))
                parsed_cases.append(parse_case(case_id, raw_record))
            else:
                for case_id, case_data in raw_record.items():
                    if isinstance(case_data, dict):
                        parsed_cases.append(parse_case(str(case_id), case_data))
                    elif isinstance(case_data, list) and not case_data:
                        parsed_cases.append(parse_case(str(case_id), {}))

    return parsed_cases

def parse_case(case_id: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Converts token/label arrays into sentences and argument spans."""
    text_lists = case_data.get("text", [])
    label_lists = case_data.get("labels", [])

    sentences = []
    relevance = {}
    premises, claims = [], []

    for i, (tokens, tags) in enumerate(zip(text_lists, label_lists)):
        sentence_str = " ".join(tokens).replace(" ,", ",").replace(" .", ".")
        sentences.append(sentence_str)

        relevance[str(i)] = any(tag != "O" for tag in tags)

        current_span, current_type = [], None

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current_span:
                    span = " ".join(current_span)
                    if current_type == "Premise":
                        premises.append(span)
                    elif current_type == "Claim":
                        claims.append(span)

                current_span = [token]
                current_type = tag.split("-")[1]

            elif tag.startswith("I-") and current_type == tag.split("-")[1]:
                current_span.append(token)

            elif tag == "O":
                if current_span:
                    span = " ".join(current_span)
                    if current_type == "Premise":
                        premises.append(span)
                    elif current_type == "Claim":
                        claims.append(span)
                current_span, current_type = [], None

        if current_span:
            span = " ".join(current_span)
            if current_type == "Premise":
                premises.append(span)
            elif current_type == "Claim":
                claims.append(span)

    return {
        "id": case_id,
        "text": sentences,
        "relevance_labels": relevance, 
        "premises": premises,
        "claims": claims,
    }

def load_relations(file_path: Path) -> List[Dict[str, Any]]:
    """Flattens the _relations.jsonl file into individual evaluation targets for Subtask 3."""
    relations_list = []
    if not file_path.exists():
        logging.error(f"\t> (!) File not found: {file_path}")
        return relations_list
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            record = json.loads(line)
            if isinstance(record, list): record = record[0]
            
            for case_id, rels in record.items():
                for idx, (head, tail, label) in enumerate(rels):
                    relations_list.append({
                        "id": f"{case_id}_{idx}",
                        "case_id": case_id,
                        "head": head,
                        "tail": tail,
                        "label": label
                    })
                    
    return relations_list