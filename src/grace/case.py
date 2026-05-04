# case.py
# ----------------------------------------------------------------------------------------
# clinical case and relations parsing for 
# · CasiMedicos https://huggingface.co/datasets/HiTZ/casimedicos-arg and for the
# · GRACE shared task dataset
# · few-shot sampling of both
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# may-2026

import src.grace.config as settings
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s", datefmt='%H:%M:%S')

def save_predictions(data: List[Dict[str, Any]], out_file: Path) -> None:
    """Saves prediction results to a JSON file."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"\t >>> Saved results to {out_file.name}")

def load_dataset(dataset: str = "grace") -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Loads all cases and relations dynamically based on the dataset (grace, casimedicos or both unified)."""
    
    # ** dataset toggle **
    if dataset == "casimedicos":
        split = settings.CASIMEDICOS_SPLITS
        is_IOB = True
    elif dataset == "unified":
        split = settings.UNIFIED_SPLITS
        is_IOB = False
    else:
        split = settings.GRACE_SPLITS
        is_IOB = False
        
    train_path = split["train"]
    test_path = split["validation"]
    
    if is_IOB:
        train_rel_path = train_path.with_name(train_path.stem.replace("_ordered", "_relations") + ".jsonl")
        test_rel_path = test_path.with_name(test_path.stem.replace("_ordered", "_relations") + ".jsonl")
        
        logging.info(f"\t\t> Loading CasiMedicos cases from: {train_path.name}")
        train_cases = load_cases_casiMedicos(train_path)
        test_cases = load_cases_casiMedicos(test_path)    
        train_relations = load_relations_casiMedicos(train_rel_path)
        test_relations = load_relations_casiMedicos(test_rel_path)
        default_origin = "CASIMEDICOS"
    else:
        logging.info(f"\t\t>> Loading GRACE-formatted cases and relations from: {train_path.name}")
        train_cases = load_cases_GRACE(train_path)
        test_cases = load_cases_GRACE(test_path)    
        train_relations = load_relations_GRACE(train_path)
        test_relations = load_relations_GRACE(test_path)
        default_origin = "GRACE"
    
    # ** improve relations data **
    train_relations = _enrich_relation_data(train_cases, train_relations, default_origin)
    test_relations = _enrich_relation_data(test_cases, test_relations, default_origin)

    # ** sort the data **
    train_cases.sort(key=lambda x: str(x.get("id", "")))
    train_relations.sort(key=lambda x: str(x.get("id", "")))
    
    
    return train_cases, train_relations, test_cases, test_relations

def sample_few_shot(train_cases: List[Dict], train_relations: List[Dict], n: int = 4, dataset: str = "grace", balanced_split: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """Samples few-shot examples from the training data, handling unified 50/50 splits."""
        
    # ** randomisation seed **
    random.seed(42)
    
    if balanced_split and dataset == "unified":
        logging.info(f"> Enforcing 50/50 GRACE/CASIMEDICOS split for few-shot cases...")
        
        grace_cases = [c for c in train_cases if c.get("origin", "").upper() == "GRACE"]
        casi_cases = [c for c in train_cases if c.get("origin", "").upper() == "CASIMEDICOS"]
        
        half_n = n // 2
        n_grace = min(half_n, len(grace_cases))
        n_casi = min(n - n_grace, len(casi_cases)) 
        
        fs_cases = random.sample(grace_cases, n_grace) + random.sample(casi_cases, n_casi)
        random.shuffle(fs_cases)
    else:
        fs_cases = random.sample(train_cases, min(n, len(train_cases)))
        
    sampled_case_ids = {str(c.get("id", "")) for c in fs_cases}
    fs_relations = [r for r in train_relations if str(r.get("case_id", "")) in sampled_case_ids]
            
    return fs_cases, fs_relations

# --- GRACE shared task parsing utilities ---

def load_cases_GRACE(file_path: Path) -> List[Dict[str, Any]]:
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
        origin = item.get("origin", metadata.get("origin", "GRACE"))
        
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
            "claims": claims,
            "origin": origin
        })

    return parsed_cases


def load_relations_GRACE(file_path: Path) -> List[Dict[str, Any]]:
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
                    "text": item.get('text', []),
                    "head": entity_map[arg1_id],
                    "tail": entity_map[arg2_id],
                    "label": rel.get("relation_type")
                })
                    
    return relations_list

# --- casiMedicos-arg case parsing utilities ---

def load_cases_casiMedicos(file_path: Path) -> List[Dict[str, Any]]:
    """Loads and parses BIO-tagged clinical cases, supporting both .json and .jsonl."""
    logging.info(f"> Loading cases from {file_path.name}")
    parsed_cases = []

    if not file_path.exists():
        logging.error(f"\t (!) File not found: {file_path}")
        return parsed_cases

    # --- standard .json files ---
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, dict):
                data = [{"id": k, **v} if isinstance(v, dict) else v for k, v in data.items()]
            elif not isinstance(data, list):
                data = [data]
                
            for raw_record in data:
                if "id" in raw_record or "case_id" in raw_record:
                    case_id = str(raw_record.get("id", raw_record.get("case_id")))
                    parsed_cases.append(parse_case_casiMedicos(case_id, raw_record))
                else:
                    for case_id, case_data in raw_record.items():
                        if isinstance(case_data, dict):
                            parsed_cases.append(parse_case_casiMedicos(str(case_id), case_data))
                        elif isinstance(case_data, list) and not case_data:
                            parsed_cases.append(parse_case_casiMedicos(str(case_id), {}))
                            
    # --- line-by-line .jsonl files ---
    else:
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
    premises = []
    claims = []
    claim_counter = 1 # Counter to assign IDs to claims

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
                        claims.append({"id": str(claim_counter), "text": span})
                        claim_counter += 1

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
                        claims.append({"id": str(claim_counter), "text": span})
                        claim_counter += 1
                current_span, current_type = [], None

        if current_span:
            span = " ".join(current_span)
            if current_type == "Premise":
                premises.append(span)
            elif current_type == "Claim":
                claims.append({"id": str(claim_counter), "text": span})
                claim_counter += 1

    origin = case_data.get("origin", "CASIMEDICOS")
    return {
        "id": case_id,
        "text": sentences,
        "relevance_labels": relevance, 
        "premises": premises,
        "claims": claims,
        "origin": origin
    }


def load_relations_casiMedicos(file_path: Path) -> List[Dict[str, Any]]:
    """Flattens the relations file into individual evaluation targets, supporting .json and .jsonl."""
    relations_list = []
    if not file_path.exists():
        logging.error(f"\t> (!) File not found: {file_path}")
        return relations_list
        
    # --- standard .json files ---
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict): data = [data]
            for record in data:
                if isinstance(record, list): record = record
                for case_id, rels in record.items():
                    for idx, (head, tail, label) in enumerate(rels):
                        relations_list.append({
                            "id": f"{case_id}_{idx}",
                            "case_id": case_id,
                            "head": head.strip(),
                            "tail": tail.strip(),
                            "label": label
                        })
                        
    # --- line-by-line .jsonl files ---
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                record = json.loads(line)
                if isinstance(record, list): record = record
                
                for case_id, rels in record.items():
                    for idx, (head, tail, label) in enumerate(rels):
                        relations_list.append({
                            "id": f"{case_id}_{idx}",
                            "case_id": case_id,
                            "head": head.strip(),
                            "tail": tail.strip(),
                            "label": label
                        })
                    
    return relations_list

# ** unified: origin of data **

def _enrich_relation_data(cases: List[Dict], relations: List[Dict], default_origin: str) -> List[Dict]:
    """Adds full case context/dataset origin to relations data."""
    case_map = {str(c["id"]): c for c in cases}
    enriched_relations = []
    
    for rel in relations:
        case_id = str(rel.get("case_id"))
        parent_case = case_map.get(case_id, {})
        
        # ** parent case data **
        rel["text"] = parent_case.get("text", rel.get("text", []))
        rel["origin"] = parent_case.get("origin", default_origin)
        
        enriched_relations.append(rel)
        
    return enriched_relations
