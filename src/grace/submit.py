# submit.py
# ----------------------------------------------------------------------------------------
# submission file compiler for predictions
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
import logging
from pathlib import Path
from .post import _extract_json_block, find_span, _load_preds
logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

# --------------- global submission compilation

def submit_global(original_json_path: Path, global_preds_path: Path, output_path: Path):
    """Compiles the final submission file from one-step global inference predictions."""
    logging.info(f"> Compiling final submission file from [GLOBAL] predictions...")
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        if isinstance(cases, dict): cases = [cases]
        
    # [global predictions]
    preds = {}
    if global_preds_path and global_preds_path.exists():
        with open(global_preds_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                preds[str(item["id"])] = item.get("predictions", item.get("prediction", ""))
                
    for case in cases:
        case_id = str(case["id"])
        raw_text = case.get("raw_text", "")
        if "predictions" not in case:
            case["predictions"] = {"sentence_relevancy": [], "entities": [], "relations": []}
            
        # Aggressive ID matching (por si el ID guardado tiene prefijos/sufijos)
        raw_pred = preds.get(case_id)
        if not raw_pred:
            for k, v in preds.items():
                if case_id in str(k): 
                    raw_pred = v
                    break
        if not raw_pred:
            raw_pred = ""
        
        # [Robust String JSON Unpacking]
        # Bucle para desempaquetar JSONs dentro de JSONs o bloques markdown
        while isinstance(raw_pred, str):
            raw_pred = raw_pred.strip()
            if not raw_pred: 
                break
                
            ext = _extract_json_block(raw_pred)
            to_parse = ext if ext else raw_pred
            
            try: 
                raw_pred = json.loads(to_parse)
            except Exception:
                # Si falla, salimos del bucle con lo que tengamos
                break
                
        pred_dict = raw_pred if isinstance(raw_pred, dict) else {}

        # ** S1: relevance **
        num_sentences = len(case.get("metadata", {}).get("context_sentences", []))
        rel_list = pred_dict.get("sentence_relevancy", [])
        
        final_relevancy = []
        for i in range(num_sentences):
            if i < len(rel_list):
                val = str(rel_list[i]).lower()
                final_relevancy.append("relevant" if val == "relevant" else "not-relevant")
            else:
                final_relevancy.append("not-relevant") # fallback: for missing sentences
        case["predictions"]["sentence_relevancy"] = final_relevancy
        
        # ** S2: entities **
        pred_entities = []
        local_to_global_id = {}
        
        # > premises (ÚNICAMENTE extrae lo que predice el modelo)
        for idx, p in enumerate(pred_dict.get("premises", [])):
            p_text = p.get("text", "").strip()
            p_local = str(p.get("local_id", f"p{idx+1}"))
            if not p_text: continue
            
            start_idx, actual_span = find_span(raw_text, p_text)
            new_id = f"pred_p_{idx+1}"
            local_to_global_id[p_local] = new_id  # "p1" to "pred_p_1"
            
            pred_entities.append({
                "id": new_id,
                "text": actual_span if start_idx != -1 else p_text,
                "start": start_idx,
                "end": start_idx + len(actual_span) if start_idx != -1 else -1,
                "type": "Premise"
            })
            
        case["predictions"]["entities"] = pred_entities
        
        # ** S3: relations **
        pred_relations = []
        for idx, r in enumerate(pred_dict.get("relations", [])):
            p_local = str(r.get("premise_id", ""))
            c_id = str(r.get("claim_id", ""))
            rel_type = str(r.get("relation_type", "")).capitalize()
            
            p_id = local_to_global_id.get(p_local)
            
            # ******** ONLY ADD IF PREMISE EXISTS AND THE RELATION IS VALID ********
            if p_id and c_id and rel_type in ["Support", "Attack"]:
                pred_relations.append({
                    "id": f"pred_rel_{idx+1}",
                    "arg1_id": p_id,
                    "arg2_id": c_id,  # (!!!) c_id should exist in entities
                    "relation_type": rel_type
                })
        case["predictions"]["relations"] = pred_relations
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
        
    logging.info(f">>> Successfully compiled [GLOBAL] submission to {output_path.name}")
    
# --------------- per-task submission compilation

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

        # ** S1: relevancy **
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

        # ** S2: entities **
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
        
        # ** S3: relations **
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