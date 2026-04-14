# case.py
# ----------------------------------------------------------------------------------------
# clinical case and relations parsing for https://huggingface.co/datasets/HiTZ/casimedicos-arg
# and for the GRACE shared tas (track 2) data
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

# --- casiMedicos-arg case parsing utilities ---

def load_cases(file_path: Path) -> List[Dict[str, Any]]:
    """Loads and parses the official shared task JSON format for S1 and S2."""
    logging.info(f"> Loading cases from {file_path.name}")
    parsed_cases = []

    if not file_path.exists():
        logging.error(f"\t> (!) File not found: {file_path}")
        return parsed_cases

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        data = [data]

    for item in data:
        case_id = item.get("id")
        metadata = item.get("metadata", {})
        annotations = item.get("annotations", {})

        # ** context sentences **
        sentences = [s.get("sentence") for s in metadata.get("context_sentences", []) if s.get("sentence") != ":"]
        
        # ** (SUBTASK 1) relevance labels **
        relevance = {}
        raw_relevance = annotations.get("sentence_relevancy", [])
        for i, status in enumerate(raw_relevance):
            relevance[str(i)] = (status == "relevant") # true:relevant, false:not_relevant, null:unlabeled

        # ** (SUBTASK 2) argumentative entities **
        premises = []
        claims = []
        entities = annotations.get("entities", [])
        
        for ent in entities:
            if ent.get("type") == "Premise":
                premises.append(ent.get("text"))
            elif ent.get("type") == "Claim":
                claims.append({"id": ent.get("id"), "text": ent.get("text")})

        # ** fully parsed train case **
        parsed_cases.append({
            "id": case_id,
            "text": sentences,
            "relevance_labels": relevance,
            "premises": premises,
            "claims": claims
        })

    return parsed_cases


def load_relations(file_path: Path) -> List[Dict[str, Any]]:
    """Flattens the official JSON format into individual evaluation targets for Subtask 3."""
    relations_list = []
    
    if not file_path.exists():
        logging.error(f"\t> (!) File not found: {file_path}")
        return relations_list
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        data = [data]

    for item in data:
        case_id = item.get("id")
        annotations = item.get("annotations", {})
        
        # ** identified argumentative entities **
        entities = annotations.get("entities", [])
        entity_map = {ent["id"]: ent["text"] for ent in entities}
        
        # ** relations **
        relations = annotations.get("relations", [])
        for rel in relations:
            arg1_id = rel.get("arg1_id")
            arg2_id = rel.get("arg2_id")
            
            if arg1_id in entity_map and arg2_id in entity_map:
                relations_list.append({
                    "id": f"{case_id}_{rel.get('id')}",
                    "case_id": case_id,
                    "head": entity_map[arg1_id],
                    "tail": entity_map[arg2_id],
                    "label": rel.get("relation_type")
                })
                    
    return relations_list

# --- casiMedicos-arg case parsing utilities ---

def load_cases_casiMedicos(file_path: Path) -> List[Dict[str, Any]]:
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
                parsed_cases.append(parse_case_casiMedicos(case_id, raw_record))
            else:
                for case_id, case_data in raw_record.items():
                    if isinstance(case_data, dict):
                        parsed_cases.append(parse_case_casiMedicos(str(case_id), case_data))
                    elif isinstance(case_data, list) and not case_data:
                        parsed_cases.append(parse_case_casiMedicos(str(case_id), {}))

    return parsed_cases

def parse_case_casiMedicos(case_id: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
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

def load_relations_casiMedicos(file_path: Path) -> List[Dict[str, Any]]:
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