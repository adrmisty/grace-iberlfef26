# submit.py
# ----------------------------------------------------------------------------------------
# submission file compiler for predictions
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
import logging
import copy
from pathlib import Path
from .clean import _extract_json_block, find_span, _load_preds

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

def _parse_json(raw_val):
    if isinstance(raw_val, (dict, list)): return raw_val
    if not isinstance(raw_val, str) or not raw_val.strip(): return {}
    ext = _extract_json_block(raw_val)
    try: return json.loads(ext if ext else raw_val)
    except: return {}

# --------------- global submission compilation

def submit_global(original_json_path: Path, global_preds_path: Path, output_path: Path):
    logging.info(f"> Compiling final submission file from [GLOBAL] predictions...")
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        if isinstance(cases, dict): cases = [cases]
        
    preds = {}
    if global_preds_path and global_preds_path.exists():
        with open(global_preds_path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                preds[str(item["id"])] = item.get("predictions", item.get("prediction", ""))
                
    for case in cases:
        case_id = str(case["id"])
        raw_text = case.get("raw_text", "")
        case["predictions"] = {"sentence_relevancy": [], "entities": [], "relations": []}
            
        raw_pred = next((v for k, v in preds.items() if case_id in str(k)), "")
        pred_dict = _parse_json(raw_pred)

        # ** S1: relevance **
        num_sentences = len(case.get("metadata", {}).get("context_sentences", []))
        rel_list = pred_dict.get("sentence_relevancy", [])
        case["predictions"]["sentence_relevancy"] = [
            "relevant" if str(r).lower() == "relevant" else "not-relevant" 
            for r in (rel_list + ["not-relevant"] * num_sentences)[:num_sentences]
        ]
        
        # ** S2: entities (Premises + Claims) **
        pred_entities = []
        local_to_global_id = {}
        
        # 1. Add Predicted Premises
        for idx, p in enumerate(pred_dict.get("premises", [])):
            p_text = p.get("text", "").strip()
            p_local = str(p.get("local_id", f"p{idx+1}"))
            if not p_text: continue
            
            start_idx, actual_span = find_span(raw_text, p_text)
            new_id = f"pred_p_{idx+1}"
            local_to_global_id[p_local] = new_id  
            pred_entities.append({
                "id": new_id, "text": actual_span if start_idx != -1 else p_text,
                "start": start_idx, "end": start_idx + len(actual_span) if start_idx != -1 else -1,
                "type": "Premise"
            })
            
        # 2. Inject Claims from raw data (since global prompt only returns IDs)
        raw_claims = case.get("claims", [e for e in case.get("annotations", {}).get("entities", []) if e.get("type") == "Claim"])
        for c in raw_claims:
            c_text = c.get("text", "")
            c_id = str(c.get("id"))
            start_idx, actual_span = find_span(raw_text, c_text)
            pred_entities.append({
                "id": c_id, "text": actual_span if start_idx != -1 else c_text,
                "start": start_idx, "end": start_idx + len(actual_span) if start_idx != -1 else -1,
                "type": "Claim"
            })
            
        case["predictions"]["entities"] = pred_entities
        
        # ** S3: relations **
        pred_relations = []
        for idx, r in enumerate(pred_dict.get("relations", [])):
            p_id, c_id = local_to_global_id.get(str(r.get("premise_id", ""))), str(r.get("claim_id", ""))
            rel_type = str(r.get("relation_type", "")).capitalize()
            
            # ** fix: avoid self-referencing relations **
            if p_id and c_id and p_id != c_id and rel_type in ["Support", "Attack"]:
                pred_relations.append({
                    "id": f"pred_rel_{idx+1}", "arg1_id": p_id, "arg2_id": c_id, "relation_type": rel_type
                })
        case["predictions"]["relations"] = pred_relations
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logging.info(f">>> Successfully compiled [GLOBAL] submission to {output_path.name}")
    

# --------------- per-task submission compilation

def submit(original_json_path: Path, s1_path: Path, s2_path: Path, s3_path: Path, output_path: Path):
    logging.info(f"> Compiling final submission file...")
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        if isinstance(gold_data, dict): gold_data = [gold_data]
    cases = copy.deepcopy(gold_data)

    s1_preds, s2_preds, s3_preds = _load_preds(s1_path), _load_preds(s2_path), _load_preds(s3_path)

    for case in cases:
        case_id, raw_text = case["id"], case.get("raw_text", "")
        case["predictions"] = {"sentence_relevancy": [], "entities": [], "relations": []}

        # ** S1: relevancy **
        s1_dict = _parse_json(s1_preds.get(case_id, {}))
        num_sents = len(case.get("metadata", {}).get("context_sentences", []))
        if "sentence_relevancy" in s1_dict:
            case["predictions"]["sentence_relevancy"] = ["relevant" if r == "relevant" else "not-relevant" for r in s1_dict["sentence_relevancy"]][:num_sents]
        else:
            case["predictions"]["sentence_relevancy"] = ["relevant" if str(s1_dict.get(str(i), s1_dict.get(i, False))).lower() in ["true", "1", "yes"] else "not-relevant" for i in range(num_sents)]

        # ** S2: entities (already parses both Premises & Claims) **
        s2_dict = _parse_json(next((v for k, v in s2_preds.items() if case_id in str(k)), {}))
        pred_entities = []
        ent_counter = 1
        
        for ent_type, key in [("Premise", "premises"), ("Claim", "claims")]:
            for item in s2_dict.get(key, s2_dict.get(key.capitalize(), [])):
                txt = item.get("text", "").strip() if isinstance(item, dict) else str(item).strip()
                if not txt: continue
                e_id = str(item.get("id")) if isinstance(item, dict) and "id" in item else f"pred_{ent_type[0].lower()}{ent_counter}"
                start_idx, actual_span = find_span(raw_text, txt)
                
                pred_entities.append({
                    "id": e_id, "text": actual_span if start_idx != -1 else txt,
                    "start": start_idx, "end": start_idx + len(actual_span) if start_idx != -1 else -1, "type": ent_type
                })
                ent_counter += 1
        case["predictions"]["entities"] = pred_entities
        
        # ** S3: relations **
        pred_relations = []
        gold_ents = {e["id"]: e["text"] for e in case.get("annotations", {}).get("entities", [])}

        for gold_rel in case.get("annotations", {}).get("relations", []):
            rel_id = gold_rel["id"]
            p_obj = _parse_json(next((v for k, v in s3_preds.items() if str(rel_id) in str(k)), {}))
            label = p_obj.get("label", "").strip().capitalize() if isinstance(p_obj, dict) else str(p_obj).strip().capitalize()
            if label not in ["Support", "Attack"]: continue

            def find_pred_id(txt):
                return next((e["id"] for e in pred_entities if e["text"] == txt or txt in e["text"] or e["text"] in txt), None)

            p1_id, p2_id = find_pred_id(gold_ents.get(gold_rel["arg1_id"])), find_pred_id(gold_ents.get(gold_rel["arg2_id"]))
            
            # ** fix: avoid self-referencing relations **
            if p1_id and p2_id and p1_id != p2_id:
                pred_relations.append({"id": rel_id, "arg1_id": p1_id, "arg2_id": p2_id, "relation_type": label})
                
        case["predictions"]["relations"] = pred_relations

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logging.info(f">>> Successfully compiled standard submission to {output_path.name}")
    
    _patch_s3_gold(cases, gold_data, s3_preds, output_path)

def _patch_s3_gold(cases_data: list, gold_data: list, s3_preds: dict, output_path: Path):
    patched_cases = copy.deepcopy(cases_data)
    gold_map = {item["id"]: item for item in gold_data}

    for item in patched_cases:
        g_case = gold_map.get(item["id"])
        if not g_case: continue

        item["predictions"]["entities"] = g_case["annotations"]["entities"]
        
        patched_rels = []
        for grel in g_case["annotations"]["relations"]:
            rel_id = grel["id"]
            p_obj = _parse_json(next((v for k, v in s3_preds.items() if str(rel_id) in str(k)), {}))
            lbl = p_obj.get("label", "").strip().capitalize() if isinstance(p_obj, dict) else str(p_obj).strip().capitalize()
            
            if lbl in ["Support", "Attack"]:
                patched_rels.append({
                    "id": rel_id, "arg1_id": grel["arg1_id"], "arg2_id": grel["arg2_id"], "relation_type": lbl
                })
        item["predictions"]["relations"] = patched_rels

    gold_out_path = output_path.with_name(output_path.stem + "_s3_gold.json")
    with open(gold_out_path, "w", encoding="utf-8") as f:
        json.dump(patched_cases, f, ensure_ascii=False, indent=2)
    logging.info(f">>> Successfully compiled GOLD-patched S3 submission to {gold_out_path.name}")