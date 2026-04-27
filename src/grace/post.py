# post.py
# ----------------------------------------------------------------------------------------
# post-processing utils
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

from collections import defaultdict
import json
import logging
import re
from pathlib import Path
from src.case import load_cases, load_relations

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def _load_preds(path: Path):
    """Loads predictions. Keys are Case IDs for S1/S2, and Relation IDs for S3."""
    if not path or not path.exists(): 
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return {item["id"]: item["prediction"] for item in json.load(f)}

def find_span(raw_text, span):
    """Finds character offsets for a text span within the raw clinical text."""
    span = span.strip().strip('"').strip("'")
    if not span: return -1, span
        
    start = raw_text.find(span)
    if start != -1: return start, span

    pattern = re.escape(span).replace(r'\ ', r'\s+')
    match = re.search(pattern, raw_text)
    if match: return match.start(), match.group()
    return -1, span

# submission compilation

def submit(original_json_path: Path, s1_path: Path, s2_path: Path, s3_path: Path, output_path: Path):
    logging.info(f"> Compiling final submission file...")
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        if isinstance(cases, dict): cases = [cases]

    s1_preds = _load_preds(s1_path)
    s2_preds = _load_preds(s2_path)
    s3_preds = _load_preds(s3_path)

    for case in cases:
        case_id = case["id"]
        raw_text = case.get("raw_text", "")
        if "predictions" not in case:
            case["predictions"] = {"sentence_relevancy": [], "entities": [], "relations": []}

        # --- SUBTASK 1: Relevance ---
        s1_raw = s1_preds.get(case_id, {})
        s1_dict = {}
        if isinstance(s1_raw, dict): 
            s1_dict = s1_raw
        elif isinstance(s1_raw, str):
            ext = _extract_json_block(s1_raw)
            try: s1_dict = json.loads(ext if ext else s1_raw)
            except: s1_dict = {}

        if not isinstance(s1_dict, dict):
            s1_dict = {}

        num_sentences = len(case.get("metadata", {}).get("context_sentences", []))
        relevancy_list = []
        for i in range(num_sentences):
            val = s1_dict.get(str(i), s1_dict.get(i, False))
            if isinstance(val, str): 
                val = val.lower() in ["true", "1", "yes"]
            relevancy_list.append("relevant" if val else "not-relevant")
        case["predictions"]["sentence_relevancy"] = relevancy_list

        # --- SUBTASK 2: Entities ---
        pred_entities = []
        ent_counter = 1
        
        s2_raw = s2_preds.get(case_id)
        if not s2_raw:
            for k, v in s2_preds.items():
                if case_id in str(k): s2_raw = v; break
                
        s2_dict = {}
        if isinstance(s2_raw, dict): 
            s2_dict = s2_raw
        elif isinstance(s2_raw, str):
            ext = _extract_json_block(s2_raw)
            try: s2_dict = json.loads(ext if ext else s2_raw)
            except: s2_dict = {}

        if isinstance(s2_dict, list) and len(s2_dict) > 0:
            s2_dict = s2_dict

        if isinstance(s2_dict, dict):
            premises = s2_dict.get("premises", s2_dict.get("Premises", []))
            claims = s2_dict.get("claims", s2_dict.get("Claims", []))

            for p in premises:
                text_span = p.get("text", "") if isinstance(p, dict) else str(p)
                text_span = text_span.strip()
                if not text_span: continue
                
                start_idx, actual_span = find_span(raw_text, text_span)
                pred_entities.append({
                    "id": f"pred_e{ent_counter}",
                    "text": actual_span if start_idx != -1 else text_span,
                    "start": start_idx,
                    "end": start_idx + len(actual_span) if start_idx != -1 else -1,
                    "type": "Premise"
                })
                ent_counter += 1

            for c in claims:
                text_span = c.get("text", "") if isinstance(c, dict) else str(c)
                text_span = text_span.strip()
                if not text_span: continue
                
                c_id = str(c.get("id")) if isinstance(c, dict) and "id" in c else f"pred_c{ent_counter}"
                
                start_idx, actual_span = find_span(raw_text, text_span)
                pred_entities.append({
                    "id": c_id,
                    "text": actual_span if start_idx != -1 else text_span,
                    "start": start_idx,
                    "end": start_idx + len(actual_span) if start_idx != -1 else -1,
                    "type": "Claim"
                })
                ent_counter += 1

        case["predictions"]["entities"] = pred_entities
        
        # --- SUBTASK 3: Relations ---
        pred_relations = []
        gold_rels = case.get("annotations", {}).get("relations", [])
        gold_ents = {e["id"]: e["text"] for e in case.get("annotations", {}).get("entities", [])}

        for gold_rel in gold_rels:
            rel_id = gold_rel["id"]
            
            pred_label_obj = s3_preds.get(rel_id)
            if not pred_label_obj:
                for k, v in s3_preds.items():
                    if str(k).endswith(str(rel_id)) or str(rel_id) in str(k):
                        pred_label_obj = v; break
            
            if isinstance(pred_label_obj, str):
                ext = _extract_json_block(pred_label_obj)
                try: pred_label_obj = json.loads(ext if ext else pred_label_obj)
                except: pass
            
            if not pred_label_obj: continue
            label = pred_label_obj.get("label", "") if isinstance(pred_label_obj, dict) else str(pred_label_obj)
            if not label or label.lower() == "nan": continue

            def find_pred_id(txt):
                for ent in case["predictions"]["entities"]:
                    if ent["text"] == txt or txt in ent["text"] or ent["text"] in txt:
                        return ent["id"]
                return None

            p1_id = find_pred_id(gold_ents.get(gold_rel["arg1_id"]))
            p2_id = find_pred_id(gold_ents.get(gold_rel["arg2_id"]))

            if p1_id and p2_id:
                pred_relations.append({
                    "id": rel_id, "arg1_id": p1_id, "arg2_id": p2_id,
                    "relation_type": label.strip().capitalize()
                })
        case["predictions"]["relations"] = pred_relations

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    
    patch_s3_gold(output_path, original_json_path, s3_preds)


# patch with S2 gold annotations for separate evaluation

def patch_s3_gold(submission_path: Path, gold_path: Path, s3_preds: dict):
    with open(submission_path, "r", encoding="utf-8") as f:
        sub_data = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = {item["id"]: item for item in json.load(f)}

    count = 0
    for item in sub_data:
        case_id = item.get("id")
        gold_case = gold_data.get(case_id)
        if not gold_case: continue

        item["predictions"]["entities"] = gold_case["annotations"]["entities"]
        
        patched_rels = []
        for grel in gold_case["annotations"]["relations"]:
            rel_id = grel["id"]
            
            p_obj = s3_preds.get(rel_id)
            if not p_obj:
                for k, v in s3_preds.items():
                    if str(k).endswith(str(rel_id)) or str(rel_id) in str(k):
                        p_obj = v; break
            
            if isinstance(p_obj, str):
                ext = _extract_json_block(p_obj)
                try: p_obj = json.loads(ext if ext else p_obj)
                except: pass

            if not p_obj: continue
            
            lbl = p_obj.get("label", "") if isinstance(p_obj, dict) else str(p_obj)
            lbl = lbl.strip().capitalize()
            if lbl in ["Support", "Attack"]:
                patched_rels.append({
                    "id": rel_id, 
                    "arg1_id": grel["arg1_id"], 
                    "arg2_id": grel["arg2_id"], 
                    "relation_type": lbl
                })
        
        item["predictions"]["relations"] = patched_rels
        count += 1

    out_path = submission_path.with_name(submission_path.stem + "_s3_gold.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)
                    

# cleaning and parsing

def _json_parse(text: str):
    """Safely attempt to parse JSON."""
    try:
        return json.loads(text)
    except Exception:
        return text


def _extract_json_block(text: str):
    """
    Extract JSON content from markdown code blocks or inline text.
    Handles:
    - ```json { ... } ```
    - ``` { ... } ```
    - fallback: first {...} block
    """
    if not isinstance(text, str):
        return None

    # Preferred: fenced code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: any JSON-like object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _list_parse(text: str):
    """
    Parse Premises/Claims from mixed-format bullet lists.
    Handles standard bullets, JSON string bullets, and YAML-style keys.
    """
    result = {"premises": [], "claims": []}

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    current = None
    current_claim = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if re.match(r"^(premisas|premises)", line, re.IGNORECASE):
            current = "premises"
            continue

        if re.match(r"^(afirmaciones|claims)", line, re.IGNORECASE):
            current = "claims"
            continue

        if not current:
            continue

        line_clean = line.lstrip("-*• \t").strip()
        
        if not line_clean or line_clean.lower() == "nan":
            continue

        if current == "premises":
            content = line_clean.strip('",\'')
            if content:
                result["premises"].append(content)

        elif current == "claims":
            # 1. Inline JSON parsing (e.g., `- {"id": 1, "text": "..."}`)
            if line_clean.startswith("{") and line_clean.endswith("}"):
                try:
                    json_str = line_clean.replace("'", '"')
                    parsed_claim = json.loads(json_str)
                    
                    if isinstance(parsed_claim, dict) and "text" in parsed_claim:
                        result["claims"].append({
                            "id": str(parsed_claim.get("id", "")),
                            "text": str(parsed_claim["text"]).strip('",\'')
                        })
                    continue
                except json.JSONDecodeError:
                    pass

            # 2. YAML parsing: Extract 'id: 1'
            if line_clean.lower().startswith("id:"):
                id_match = re.search(r"id:\s*([a-zA-Z0-9_]+)", line_clean, re.IGNORECASE)
                if id_match:
                    current_claim["id"] = id_match.group(1)
                continue

            # 3. YAML parsing
            if line_clean.lower().startswith("text:"):
                text_match = re.search(r"text:\s*(.*)", line_clean, re.IGNORECASE)
                if text_match:
                    extracted_text = text_match.group(1).strip().strip('",\'')
                    result["claims"].append({
                        "id": current_claim.get("id", ""),
                        "text": extracted_text
                    })
                    current_claim = {}
                continue

            result["claims"].append({
                "id": current_claim.get("id", ""),
                "text": line_clean.strip('",\'')
            })
            current_claim = {}

    return result if (result["premises"] or result["claims"]) else text

def clean(filepath: Path):
    """
    Normalize raw LLM outputs into structured JSON.
    """
    if not filepath.exists():
        logging.warning(f"\t(!) > File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0

    for item in data:
        pred = item.get("prediction", "")

        if not isinstance(pred, str):
            continue

        if "</think>" in pred:
            pred = pred.split("</think>")[-1].strip()
            cleaned_count += 1

        pred = pred.replace("\\n", "\n").strip()

        json_block = _extract_json_block(pred)
        if json_block:
            parsed = _json_parse(json_block)
            if isinstance(parsed, dict) and ("premises" in parsed or "claims" in parsed):
                item["prediction"] = parsed
                continue

        parsed = _json_parse(pred)
        if isinstance(parsed, dict) and ("premises" in parsed or "claims" in parsed):
            item["prediction"] = parsed
            cleaned_count += 1
            continue

        if re.search(r"(Premisas|Premises)", pred, re.IGNORECASE):
            item["prediction"] = _list_parse(pred)
            cleaned_count += 1
            continue

        item["prediction"] = pred

    with open(filepath.with_suffix(".clean.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f">>> Cleaned {cleaned_count} entries in {filepath.name}")