# post.py
# ----------------------------------------------------------------------------------------
# post-processing utils
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")                    

# ------------------------------------ cleaning and so on

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
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

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
