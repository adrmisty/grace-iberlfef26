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
            
        # headers
        if re.match(r"^(premisas|premises)(:)?", line, re.IGNORECASE):
            current_list = "premises"
            continue
        elif re.match(r"^(afirmaciones|claims)(:)?", line, re.IGNORECASE):
            current_list = "claims"
            continue
            
        # extract lines
        if current_list:
            line = line.lstrip("- ").strip()
            if not line or "nan" in line.lower():
                continue
            line = line.strip("[]").strip()

            result[current_list].append(line)

    block_len = 150
    for k in ["premises", "claims"]:
        if len(result[k]) == 1 and len(result[k]) > block_len:
            result[k] = _entity_parse(result[k])

    return result if (result["premises"] or result["claims"]) else text