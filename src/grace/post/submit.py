# post.py
# ----------------------------------------------------------------------------------------
# post-processing utils & submission file compiler
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# may-2026

import json
import logging
import re
import copy
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

# ------------------------------------ cleaning and parsing

def _json_parse(text: str):
    try: return json.loads(text)
    except Exception: return text

def _extract_json_block(text: str):
    if not isinstance(text, str): return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

def _parse_json(raw_val):
    """Robust json parser to safely unpack strings/markdown blocks."""
    if isinstance(raw_val, (dict, list)): return raw_val
    if not isinstance(raw_val, str) or not raw_val.strip(): return {}
    
    raw_val = raw_val.strip()
    if raw_val.startswith('```json'): raw_val = raw_val[7:]
    if raw_val.startswith('```'): raw_val = raw_val[3:]
    if raw_val.endswith('```'): raw_val = raw_val[:-3]
    raw_val = raw_val.strip()
    
    try: return json.loads(raw_val)
    except: 
        ext = _extract_json_block(raw_val)
        try: return json.loads(ext if ext else raw_val)
        except:
            try: return json.loads(raw_val.replace("'", '"'))
            except: return {}

def _list_parse(text: str):
    result = {"premises": [], "claims": []}
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    current = None
    current_claim = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line: continue

        if re.match(r"^(premisas|premises)", line, re.IGNORECASE):
            current = "premises"; continue
        if re.match(r"^(afirmaciones|claims)", line, re.IGNORECASE):
            current = "claims"; continue
        if not current: continue

        line_clean = line.lstrip("-*• \t").strip()
        if not line_clean or line_clean.lower() == "nan": continue

        if current == "premises":
            content = line_clean.strip('",\'')
            if content: result["premises"].append(content)

        elif current == "claims":
            if line_clean.startswith("{") and line_clean.endswith("}"):
                try:
                    parsed_claim = json.loads(line_clean.replace("'", '"'))
                    if isinstance(parsed_claim, dict) and "text" in parsed_claim:
                        result["claims"].append({"id": str(parsed_claim.get("id", "")), "text": str(parsed_claim["text"]).strip('",\'')})
                    continue
                except: pass

            if line_clean.lower().startswith("id:"):
                id_match = re.search(r"id:\s*([a-zA-Z0-9_]+)", line_clean, re.IGNORECASE)
                if id_match: current_claim["id"] = id_match.group(1)
                continue

            if line_clean.lower().startswith("text:"):
                text_match = re.search(r"text:\s*(.*)", line_clean, re.IGNORECASE)
                if text_match:
                    result["claims"].append({"id": current_claim.get("id", ""), "text": text_match.group(1).strip().strip('",\'')})
                    current_claim = {}
                continue

            result["claims"].append({"id": current_claim.get("id", ""), "text": line_clean.strip('",\'')})
            current_claim = {}

    return result if (result["premises"] or result["claims"]) else text

def clean(filepath: Path):
    if not filepath.exists(): return
    with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
    cleaned_count = 0
    for item in data:
        pred = item.get("prediction", "")
        if not isinstance(pred, str): continue
        if "</think>" in pred:
            pred = pred.split("</think>")[-1].strip()
            cleaned_count += 1
        pred = pred.replace("\\n", "\n").strip()

        json_block = _extract_json_block(pred)
        if json_block:
            parsed = _json_parse(json_block)
            if isinstance(parsed, dict) and ("premises" in parsed or "claims" in parsed):
                item["prediction"] = parsed; continue

        parsed = _json_parse(pred)
        if isinstance(parsed, dict) and ("premises" in parsed or "claims" in parsed):
            item["prediction"] = parsed; cleaned_count += 1; continue

        if re.search(r"(Premisas|Premises)", pred, re.IGNORECASE):
            item["prediction"] = _list_parse(pred); cleaned_count += 1; continue
        item["prediction"] = pred

    with open(filepath.with_suffix(".clean.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_preds(path):
    path_obj = Path(path)
    if not path_obj.exists(): 
        logging.warning(f"\t(!) Missing prediction file: {path_obj.name}")
        return {}
    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict): data = [data]
            return {str(item.get("id", "")): item.get("prediction", item.get("predictions", "")) for item in data if isinstance(item, dict)}
    except Exception as e:
        logging.error(f"\t(!) Failed to load {path_obj.name}: {e}")
        return {}

def find_span(raw_text, span):
    span = span.strip().strip('"').strip("'")
    if not span: return -1, span
    start = raw_text.find(span)
    if start != -1: return start, span
    
    # substr.
    for chunk in span.split(". "):
        start = raw_text.find(chunk.strip())
        if start != -1: return start, chunk.strip()
        
    return -1, span

def _get_raw_text(case):
    raw = case.get("raw_text", "")
    if not raw:
        text_data = case.get("text", "")
        raw = " ".join(text_data) if isinstance(text_data, list) else str(text_data)
    return raw

def _extract_s3_label(p_obj):
    """Extracts the relation label supporting Pydantic Strict schemas and legacy formats."""
    if isinstance(p_obj, list) and len(p_obj) > 0: p_obj = p_obj[0]
    if isinstance(p_obj, dict):
        if "relations" in p_obj and isinstance(p_obj["relations"], list) and len(p_obj["relations"]) > 0:
            return str(p_obj["relations"][0].get("relation_type", ""))
        return str(p_obj.get("label", p_obj.get("relation_type", "")))
    return str(p_obj)

# --------------- GLOBAL submission compilation

def submit_global(original_json_path: Path, global_preds_path: Path, output_path: Path):
    with open(original_json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        if isinstance(cases, dict): cases = [cases]
        
    preds = _load_preds(global_preds_path)
    if not preds: return
                
    for case in cases:
        case_id = str(case["id"])
        raw_text = _get_raw_text(case)
        case["predictions"] = {"sentence_relevancy": [], "entities": [], "relations": []}
            
        raw_pred = next((v for k, v in preds.items() if case_id in str(k) or str(k) in case_id), "")
        pred_dict = _parse_json(raw_pred)

        num_sentences = len(case.get("metadata", {}).get("context_sentences", []))
        if num_sentences == 0: num_sentences = len(case.get("text", []))
        
        rel_list = pred_dict.get("sentence_relevancy", [])
        case["predictions"]["sentence_relevancy"] = [
            "relevant" if str(r).lower() == "relevant" else "not-relevant" 
            for r in (rel_list + ["not-relevant"] * num_sentences)[:num_sentences]
        ]
        
        pred_entities = []
        local_to_global_id = {}
        
        for idx, p in enumerate(pred_dict.get("premises", [])):
            p_text = p.get("text", "").strip()
            p_local = str(p.get("local_id", f"p{idx+1}"))
            if not p_text: continue
            
            start_idx, actual_span = find_span(raw_text, p_text)
            new_id = f"pred_p_{idx+1}"
            local_to_global_id[p_local] = new_id  
            pred_entities.append({"id": new_id, "text": actual_span if start_idx != -1 else p_text, "start": start_idx, "end": start_idx + len(actual_span) if start_idx != -1 else -1, "type": "Premise"})
            
        raw_claims = case.get("claims", [])
        if not raw_claims and "annotations" in case:
            raw_claims = [e for e in case["annotations"].get("entities", []) if e.get("type") == "Claim"]
            
        for c in raw_claims:
            c_text = c.get("text", "")
            c_id = str(c.get("id"))
            start_idx, actual_span = find_span(raw_text, c_text)
            pred_entities.append({"id": c_id, "text": actual_span if start_idx != -1 else c_text, "start": start_idx, "end": start_idx + len(actual_span) if start_idx != -1 else -1, "type": "Claim"})
            
        case["predictions"]["entities"] = pred_entities
        
        pred_relations = []
        for idx, r in enumerate(pred_dict.get("relations", [])):
            p_local = str(r.get("premise_id", "")).strip()
            c_id = str(r.get("claim_id", "")).strip()
            rel_type = str(r.get("relation_type", "")).capitalize()
            
            p_id = local_to_global_id.get(p_local)
            if p_id and c_id and p_id != c_id and rel_type in ["Support", "Attack"]:
                pred_relations.append({"id": f"pred_rel_{idx+1}", "arg1_id": p_id, "arg2_id": c_id, "relation_type": rel_type})
        case["predictions"]["relations"] = pred_relations
        
    with open(output_path, 'w', encoding='utf-8') as f: json.dump(cases, f, ensure_ascii=False, indent=2)
    logging.info(f">>> [GLOBAL] submission to {output_path.name}")
    

# --------------- SPLIT (PER-TASK) submission compilation


def submit(original_json_path: Path, s1_path: Path, s2_path: Path, s3_path: Path, output_path: Path):
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        if isinstance(gold_data, dict): gold_data = [gold_data]
    cases = copy.deepcopy(gold_data)

    s1_preds = _load_preds(s1_path)
    s2_preds = _load_preds(s2_path)
    s3_preds = _load_preds(s3_path)

    for case in cases:
        case_id = str(case["id"])
        raw_text = _get_raw_text(case)
        case["predictions"] = {"sentence_relevancy": [], "entities": [], "relations": []}

        # ** S1: relevance **
        raw_s1 = s1_preds.get(case_id, {})
        s1_dict = _parse_json(raw_s1)
        
        num_sentences = len(case.get("metadata", {}).get("context_sentences", []))
        if num_sentences == 0: num_sentences = len(case.get("text", []))
        
        rel_list = s1_dict.get("sentence_relevancy", [])
        
        if rel_list:
            # pad with not-relevant to avoid sentence mismatch
            # e.g. INFECCIOSAS_gdGMoi only predicted 3 relevance labels
            # but the fourth (punctuation mark) was ignored
            case["predictions"]["sentence_relevancy"] = [
                "relevant" if str(r).lower() == "relevant" else "not-relevant" 
                for r in (rel_list + ["not-relevant"] * num_sentences)[:num_sentences]
            ]
        
        # ** S2: entities **
        raw_s2 = s2_preds.get(case_id, {})
        s2_dict = _parse_json(raw_s2)

        pred_entities = []
        ent_counter = 1
        
        # predicted premises + injected claims
        for ent_type, key in [("Premise", "premises"), ("Claim", "claims")]:
            for item in s2_dict.get(key, s2_dict.get(key.capitalize(), [])):
                txt = item.get("text", "").strip() if isinstance(item, dict) else str(item).strip()
                if not txt: continue
                
                e_id = str(item.get("id")) if isinstance(item, dict) and "id" in item else f"pred_{ent_type[0].lower()}_{ent_counter}"
                start_idx, actual_span = find_span(raw_text, txt)
                
                pred_entities.append({
                    "id": e_id, 
                    "text": actual_span if start_idx != -1 else txt, 
                    "start": start_idx, 
                    "end": start_idx + len(actual_span) if start_idx != -1 else -1, 
                    "type": ent_type
                })
                ent_counter += 1
                
        raw_claims = case.get("claims", [])
        if not raw_claims and "annotations" in case:
            raw_claims = [e for e in case["annotations"].get("entities", []) if e.get("type") == "Claim"]
            
        existing_claim_texts = [" ".join(e["text"].strip().split()).lower() for e in pred_entities if e["type"] == "Claim"]

        for c in raw_claims:
            c_text = c.get("text", "")
            c_id = str(c.get("id"))
            
            c_norm = " ".join(c_text.strip().split()).lower()
            if c_norm not in existing_claim_texts:
                start_idx, actual_span = find_span(raw_text, c_text)
                pred_entities.append({
                    "id": c_id, 
                    "text": actual_span if start_idx != -1 else c_text, 
                    "start": start_idx, 
                    "end": start_idx + len(actual_span) if start_idx != -1 else -1, 
                    "type": "Claim"
                })
                
        case["predictions"]["entities"] = pred_entities
        
        # ** S3 relations **
        pred_relations = []
        gold_ents = {str(e["id"]): e["text"] for e in case.get("annotations", {}).get("entities", [])}

        for gold_rel in case.get("annotations", {}).get("relations", []):
            rel_id = str(gold_rel["id"])
            
            # rel_id ('INFECCIOSAS_[gdGMoi]_[FZTNTb]')
            raw_s3 = next((v for k, v in s3_preds.items() if rel_id in str(k)), None)
            if not raw_s3: continue
            
            p_obj = _parse_json(raw_s3)
            label = _extract_s3_label(p_obj).strip().capitalize()
            
            if label not in ["Support", "Attack"]: continue

            def find_pred_id(gold_txt):
                if not gold_txt: return None
                t_norm = " ".join(gold_txt.strip().split()).lower()
                for ent in pred_entities:
                    e_norm = " ".join(ent["text"].strip().split()).lower()
                    if e_norm == t_norm or t_norm in e_norm or e_norm in t_norm:
                        return ent["id"]
                return None

            p1_id = find_pred_id(gold_ents.get(str(gold_rel["arg1_id"])))
            p2_id = find_pred_id(gold_ents.get(str(gold_rel["arg2_id"])))

            if p1_id and p2_id and p1_id != p2_id:
                pred_relations.append({
                    "id": rel_id, 
                    "arg1_id": p1_id, 
                    "arg2_id": p2_id, 
                    "relation_type": label
                })
                
        case["predictions"]["relations"] = pred_relations

    with open(output_path, 'w', encoding='utf-8') as f: 
        json.dump(cases, f, ensure_ascii=False, indent=2)
        
    logging.info(f"\t>>> [S1->S2->S3] submission to {output_path.name}")
    patch_s3_gold(output_path, original_json_path, s3_preds)


def patch_s3_gold(submission_path: Path, gold_path: Path, s3_preds: dict):
    """Patches S3 predictions with gold entities from S2 and S3 labels for isolated evaluation."""
    with open(submission_path, "r", encoding="utf-8") as f: 
        sub_data = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_cases = json.load(f)
        if isinstance(gold_cases, dict): gold_cases = [gold_cases]
        gold_data = {str(item["id"]): item for item in gold_cases}

    for item in sub_data:
        g_case = gold_data.get(str(item.get("id")))
        if not g_case: continue

        item["predictions"]["entities"] = g_case.get("annotations", {}).get("entities", [])
        
        patched_rels = []
        for grel in g_case.get("annotations", {}).get("relations", []):
            rel_id = str(grel["id"])
            
            raw_s3 = next((v for k, v in s3_preds.items() if rel_id in str(k)), None)
            if not raw_s3: continue
            
            p_obj = _parse_json(raw_s3)
            lbl = _extract_s3_label(p_obj).strip().capitalize()
            
            if lbl in ["Support", "Attack"]:
                patched_rels.append({
                    "id": rel_id, 
                    "arg1_id": grel["arg1_id"], 
                    "arg2_id": grel["arg2_id"], 
                    "relation_type": lbl
                })
                
        item["predictions"]["relations"] = patched_rels

    out_path = submission_path.with_name(submission_path.stem + "_s3_gold.json")
    with open(out_path, "w", encoding="utf-8") as f: 
        json.dump(sub_data, f, ensure_ascii=False, indent=2)
    logging.info(f"\t\t>>> [S2-GOLD-patched S3] submission to {out_path.name}")