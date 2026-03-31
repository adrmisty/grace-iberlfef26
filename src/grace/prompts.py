# prompts.py
# ----------------------------------------------------------
# configurations for prompts
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import json
from typing import Dict, Any, List, Optional

# --- static prompt builders for GRACE subtasks -------------------------------------------------------------------------

SYSTEM_PROMPT: Dict[str, str] = {
    "SUBTASK_1": (
        "Eres un asistente médico experto."
        "Tu tarea es clasificar si una oración "
        "proporciona información clínica relevante (true) o no (false) "
        "para apoyar o refutar un diagnóstico o tratamiento propuesto. "
        "Responde estrictamente con un formato JSON donde las claves son el número de la oración y los valores son true o false."
    ),
    
    "SUBTASK_2": (
        "Eres un experto clínico especializado en NLP. "
        "Tu tarea es identificar y extraer los límites exactos de texto (spans) que representan "
        "PREMISES (evidencia clínica, síntomas, resultados de laboratorio) o CLAIMS (opciones de diagnóstico o tratamiento propuestas)."
        "Responde estrictamente las cadenas de texto exactas sin modificarlas y su clasificación (premise/claim)."
    ),
    
    "SUBTASK_3": (
        "Eres un especialista en razonamiento clínico. "
        "Dada una PREMISE (evidencia clínica) y una CLAIM (una opción de respuesta o diagnóstico), "
        "tu tarea es clasificar la relación direccional entre ellas. "
        "Responde únicamente con 'Support' (si la evidencia apoya la opción) "
        "o 'Attack' (si la evidencia refuta o va en contra de la opción)."
    ),
}

# --- dynamic prompt builders for GRACE subtasks -------------------------------------------------------------------------

def build_s1_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
    prompt = "AVISO IMPORTANTE: Eres un script de automatización. Genera ÚNICAMENTE la salida estructurada solicitada. Prohibido generar 'Thinking Process' o explicaciones.\n\n"
    if examples:
        prompt += "--- EJEMPLOS ---\n"
        for ex in examples:
            prompt += f"Caso clínico:\n{ex.get('text', '')}\n"
            prompt += f"Output esperado (JSON):\n{json.dumps(ex.get('relevance_labels', {}), ensure_ascii=False)}\n\n"
        prompt += "--- FIN DE EJEMPLOS ---\n\n"
    else:
        prompt += "FORMATO ESPERADO: Un diccionario JSON estricto donde las claves son los índices de las oraciones y los valores son booleanos (true/false).\nEjemplo de formato:\n{\n  \"0\": true,\n  \"1\": false\n}\n\n"
        
    prompt += "A continuación el caso clínico a clasificar:\n\n"
    sentences = case.get('text', [])
    if isinstance(sentences, list):
        for i, sent in enumerate(sentences):
            sent_str = " ".join(sent) if isinstance(sent, list) else sent
            prompt += f"[{i}] {sent_str}\n"
    else:
        prompt += f"{sentences}\n"
        
    prompt += "\nDevuelve el JSON con las clasificaciones (true/false) para cada oración."
    prompt += "\nREGLA ESTRICTA: NO escribas 'Thinking Process'. NO des explicaciones. Devuelve ÚNICAMENTE un JSON válido que empiece con '{' y termine con '}'."
    return prompt

def build_s2_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
    prompt = "AVISO IMPORTANTE: Eres un script de automatización. Genera ÚNICAMENTE la salida estructurada solicitada. Prohibido generar 'Thinking Process' o explicaciones.\n\n"
    if examples:
        prompt += "--- EJEMPLOS ---\n"
        for ex in examples:
            prompt += f"Caso clínico:\n{ex.get('text', '')}\n"
            prompt += f"Premisas extraídas:\n- " + "\n- ".join(ex.get('premises', [])) + "\n"
            prompt += f"Claims Extraídas:\n- " + "\n- ".join(ex.get('claims', [])) + "\n\n"
        prompt += "--- FIN DE EJEMPLOS ---\n\n"
    else:
        prompt += "FORMATO ESPERADO ESTRICTO:\nPremisas:\n- [span 1]\n- [span 2]\n\nAfirmaciones:\n- [span 3]\n\n"
        
    prompt += "A continuación se presenta el caso clínico:\n\n"
    text = case.get('text', '')
    if isinstance(text, list):
        text = " ".join([" ".join(s) if isinstance(s, list) else s for s in text])
        
    prompt += f"{text}\n\n"
    prompt += "Extrae los límites exactos de texto. Devuelve el resultado con el siguiente formato:\nPremisas:\n- [span 1]\n- [span 2]\n\nAfirmaciones:\n- [span 3]"
    prompt += "\nREGLA ESTRICTA: NO escribas 'Thinking Process'. NO des explicaciones. Devuelve ÚNICAMENTE las listas requeridas en el formato exacto empezando por 'Premisas:'."
    return prompt

def build_s3_prompt(relation: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
    prompt = "AVISO IMPORTANTE: Eres un script de automatización. Genera ÚNICAMENTE la salida estructurada solicitada. Prohibido generar 'Thinking Process' o explicaciones.\n\n"
    if examples:
        prompt += "--- EJEMPLOS ---\n"
        for ex in examples:
            prompt += f"Premise: \"{ex.get('head', '')}\"\n"
            prompt += f"Claim: \"{ex.get('tail', '')}\"\n"
            prompt += f"Output esperado:\n{{\n  \"label\": \"{ex.get('label', '')}\"\n}}\n\n"
        prompt += "--- FIN DE EJEMPLOS ---\n\n"
    else:
        prompt += "FORMATO ESPERADO: Un JSON estricto con la clave 'label'.\nEjemplo:\n{\n  \"label\": \"Support\"\n}\n\n"
        
    prompt += "Clasifica la siguiente relación:\n\n"
    prompt += f"Premise: \"{relation.get('head', '')}\"\n"
    prompt += f"Claim: \"{relation.get('tail', '')}\"\n"
    prompt += "\nDevuelve el JSON con la clasificación ('Support' o 'Attack')."
    return prompt