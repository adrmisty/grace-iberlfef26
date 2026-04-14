# post.py
# ----------------------------------------------------------------------------------------
# post-processing utils
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def submit(original_json_path: Path, s1_path: Path, s2_path: Path, s3_path: Path, output_path: Path):
    """Merges individual subtask predictions into the official IberLEF submission format."""
    logging.info(f"> Compiling final submission file...")
    
    if not original_json_path.exists():
        logging.error(f"\t(!) Original JSON not found: {original_json_path}")
        return

    with open(original_json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        if isinstance(cases, dict):
            cases = [cases]

    def _load_preds(path: Path):
        if not path or not path.exists(): 
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return {item["id"]: item["prediction"] for item in json.load(f)}

    def find_span(raw_text, span):
        # Strip trailing/leading punctuation and literal quotes hallucinated by the LLM
        span = span.strip().strip('"').strip("'")
        if not span:
            return -1, span
            
        # 1. Try exact match
        start = raw_text.find(span)
        if start != -1:
            return start, span

        # 2. Try regex match (handles weird spacing)
        pattern = re.escape(span).replace(r'\ ', r'\s+')
        match = re.search(pattern, raw_text)
        if match:
            return match.start(), match.group()

        return -1, span

    s1_preds = _load_preds(s1_path)
    s2_preds = _load_preds(s2_path)  # kept for compatibility (not used)
    s3_preds = _load_preds(s3_path)

    for case in cases:
        case_id = case["id"]
        raw_text = case.get("raw_text", "")
        
        if "predictions" not in case:
            case["predictions"] = {
                "sentence_relevancy": [],
                "entities": [],
                "relations": []
            }

        # ------------------------------------------------------------------
        # Subtask 1: sentence relevancy
        # ------------------------------------------------------------------
        num_sentences = len(case.get("metadata", {}).get("context_sentences", []))
        relevancy_list = []
        
        s1_dict = s1_preds.get(case_id, {})
        
        if isinstance(s1_dict, str):
            try:
                s1_dict = json.loads(s1_dict)
            except json.JSONDecodeError:
                s1_dict = {}
                
        if not isinstance(s1_dict, dict):
            s1_dict = {}

        for i in range(num_sentences):
            is_relevant = s1_dict.get(str(i), s1_dict.get(i, False))
            
            if isinstance(is_relevant, str):
                is_relevant = is_relevant.lower() in ["true", "1", "yes"]
                
            relevancy_list.append("relevant" if is_relevant else "not-relevant")
            
        case["predictions"]["sentence_relevancy"] = relevancy_list

        # ------------------------------------------------------------------
        # Subtask 2: entities (SAFE STRATEGY)
        # ------------------------------------------------------------------
        pred_entities = []
        entity_text_to_id = {} # CRITICAL for S3 mapping
        ent_counter = 1

        s2_dict = s2_preds.get(case_id, {})
        
        # Fallback if S2 is a stringified JSON
        if isinstance(s2_dict, str):
            try: s2_dict = json.loads(s2_dict)
            except: s2_dict = {}

        if isinstance(s2_dict, dict):

            # ---- Premises ----
            for text_span in s2_dict.get("premises", []):
                if not isinstance(text_span, str) or not text_span.strip():
                    continue

                start_idx, actual_span = find_span(raw_text, text_span)

                if start_idx != -1:
                    ent_id = f"pred_e{ent_counter}"
                    pred_entities.append({
                        "id": ent_id,
                        "text": actual_span,
                        "start": start_idx,
                        "end": start_idx + len(actual_span),
                        "type": "Premise"
                    })
                    entity_text_to_id[actual_span] = ent_id
                    ent_counter += 1

            # ---- Claims ----
            for item in s2_dict.get("claims", []):
                if isinstance(item, dict):
                    text_span = item.get("text", "").strip()
                    claim_id = str(item.get("id", f"pred_c{ent_counter}"))
                else:
                    text_span = str(item).strip()
                    claim_id = f"pred_c{ent_counter}"

                if not text_span:
                    continue

                start_idx, actual_span = find_span(raw_text, text_span)

                if start_idx != -1:
                    pred_entities.append({
                        "id": claim_id,  
                        "text": actual_span,
                        "start": start_idx,
                        "end": start_idx + len(actual_span),
                        "type": "Claim"
                    })
                    entity_text_to_id[actual_span] = claim_id
                    ent_counter += 1

        case["predictions"]["entities"] = pred_entities
        
        # ------------------------------------------------------------------
        # Subtask 3: relations
        # ------------------------------------------------------------------
        pred_relations = []
        
        if s3_preds:
            gold_relations = case.get("annotations", {}).get("relations", [])
            gold_entities = {e["id"]: e["text"] for e in case.get("annotations", {}).get("entities", [])}

            for gold_rel in gold_relations:
                rel_id = gold_rel["id"]

                predicted_label = None
                for key, value in s3_preds.items():
                    if key.endswith(f"_{rel_id}"):
                        predicted_label = value
                        break

                if isinstance(predicted_label, dict):
                    predicted_label = predicted_label.get("label", "")

                if not isinstance(predicted_label, str) or not predicted_label:
                    continue

                g_arg1_text = gold_entities.get(gold_rel["arg1_id"], "")
                g_arg2_text = gold_entities.get(gold_rel["arg2_id"], "")

                def find_pred_id(gold_text):
                    if not gold_text: return None
                    for ent in pred_entities:
                        if ent["text"] == gold_text:
                            return ent["id"]
                    for ent in pred_entities:
                        if gold_text in ent["text"] or ent["text"] in gold_text:
                            return ent["id"]
                    return None

                p_arg1_id = find_pred_id(g_arg1_text)
                p_arg2_id = find_pred_id(g_arg2_text)

                if p_arg1_id and p_arg2_id:
                    pred_relations.append({
                        "id": rel_id,
                        "arg1_id": p_arg1_id,   # Translated ID
                        "arg2_id": p_arg2_id,   # Translated ID
                        "relation_type": predicted_label.capitalize()
                    })

        case["predictions"]["relations"] = pred_relations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logging.info(f"\t>>> Successfully compiled submission file: {output_path.name}")    


def clean(filepath: Path):
    if not filepath.exists():
        logging.warning(f"\t(!) > File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0

    for item in data:
        pred = item.get("prediction", "")

        # <think> hallucinations
        if "</think>" in pred:
            pred = pred.split("</think>").strip()
            cleaned_count += 1

        # newlines
        pred = pred.replace("\\n", "\n").strip()

        parsed = _json_parse(pred)

        if isinstance(parsed, dict):
            item["prediction"] = parsed
            continue

        # parse premises and claims into list
        if re.search(r"(Premisas|Premises)", pred, re.IGNORECASE) and re.search(r"(Afirmaciones|Claims)", pred, re.IGNORECASE):
            item["prediction"] = _list_parse(pred)
            continue

        # cleaned string
        item["prediction"] = pred

    with open(filepath.with_suffix(".clean.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f">>> Cleaned {cleaned_count} entries in {filepath.name}")
    
# --- aux -------------------------------------------------------------------------

def _json_parse(text):
    """Try to parse a JSON string safely."""
    try:
        return json.loads(text)
    except Exception:
        return text

def _entity_parse(text):
    """Parse full text into list of spans of argument entity strings."""
    text = re.sub(r"\s+", " ", text).strip()

    # brackets: [ premise... ]
    bracket_spans = re.findall(r"\[\s*(.*?)\s*\]", text)
    if bracket_spans:
        spans = [s.strip() for s in bracket_spans if len(s.strip()) > 3]
        if spans:
            return spans

    # bullets : - premise....
    if "-" in text:
        parts = re.split(r"\s*-\s+", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 5]
        if len(parts) > 1:
            return parts

    # plain sentences
    sentences = re.split(r'(?<=[\.\?\!])\s+(?=[A-ZÁÉÍÓÚÑ])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    return sentences

def _list_parse(text):
    """Convert raw text into structured JSON with span-level premises."""
    result = {"premises": [], "claims": []}
    text = text.replace("\\n", "\n")
    
    current_list = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        if re.match(r"^(premisas|premises)", line, re.IGNORECASE):
            current_list = "premises"
            continue
        elif re.match(r"^(afirmaciones|claims)", line, re.IGNORECASE):
            current_list = "claims"
            continue
            
        if current_list:
            line = line.lstrip("- *").strip()
            if not line or "nan" in line.lower():
                continue

            if current_list == "premises":
                result["premises"].append(line)
                
            elif current_list == "claims":
                if line.startswith("{") and line.endswith("}"):
                    try:
                        claim_obj = json.loads(line)
                        result["claims"].append(claim_obj)
                    except json.JSONDecodeError:
                        result["claims"].append({"id": "", "text": line})
                else:
                    result["claims"].append({"id": "", "text": line})

    return result if (result["premises"] or result["claims"]) else text