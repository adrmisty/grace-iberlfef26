# prompts.py
# ----------------------------------------------------------
# configurations for prompts
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import json
from typing import Dict, Any, List, Optional

# --- static prompt builders -------------------------------------------------------------------------

SYSTEM_PROMPT: Dict[str, str] = {
    "SUBTASK_1": (
        "Eres un experto clínico. Tu tarea es evaluar una lista numerada de oraciones de un caso clínico. "
        "Debes determinar si cada oración contiene evidencia médica RELEVANTE (síntomas, historial, pruebas) o "
        "si es irrelevante (texto de relleno, saludos, la pregunta final).\n"
        "REGLA ESTRICTA: Devuelve ÚNICAMENTE un objeto JSON donde las claves son los índices de las oraciones y los valores son booleanos (true/false)."
    ),
    
    "SUBTASK_2": (
        "Eres un experto clínico especializado en NLP. Tu tarea es extraer fragmentos exactos de texto (spans). "
        "REGLAS CLAVE:\n"
        "1. Premise: Hechos descriptivos del paciente (síntomas, antecedentes, exploración física). "
        "Extrae la frase exacta del texto.\n"
        "2. Claim: TODAS las opciones de respuesta múltiple al final del caso. "
        "Cada opción es un claim distinto. Debes extraer su ID (1, 2, 3, etc.) y su texto exacto.\n"
        "3. EXCLUSIONES: NO extraigas la pregunta en sí (ej. '¿Qué diagnóstico...?').\n"
        "REGLA ESTRICTA: Devuelve ÚNICAMENTE un objeto JSON válido con las claves 'premises' (lista de strings) y 'claims' (lista de objetos con 'id' y 'text')."
    ),
    
    "SUBTASK_3": (
        "Eres un razonador clínico. Se te dará una PREMISE (un hecho del paciente) y un CLAIM (una posible respuesta/diagnóstico). "
        "Tu tarea es determinar la relación argumentativa entre ellos:\n"
        "- 'Support': La premisa apoya, confirma o es consistente con el claim.\n"
        "- 'Attack': La premisa contradice, descarta o hace improbable el claim.\n"
        "REGLA ESTRICTA: Devuelve ÚNICAMENTE un JSON válido con la clave 'label'."
    )
}

# --- dynamic prompt builders -------------------------------------------------------------------------

def build_s1_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
    prompt = ""
    
    if examples:
        prompt += "--- EJEMPLOS ---\n"
        for ex in examples:
            prompt += "Oraciones:\n"
            for i, sent in enumerate(ex.get('text', [])):
                prompt += f"[{i}] {sent}\n"
            prompt += f"Salida esperada:\n{{\n"
            labels = ex.get('relevance_labels', {})
            items = [f'  "{k}": {"true" if v else "false"}' for k, v in labels.items()]
            prompt += ",\n".join(items) + "\n}\n\n"
        prompt += "--- FIN DE EJEMPLOS ---\n\n"
        
    prompt += "Evalúa el siguiente caso clínico:\nOraciones:\n"
    for i, sent in enumerate(case.get('text', [])):
        prompt += f"[{i}] {sent}\n"
        
    prompt += "\nGenera el JSON de salida:"
    return prompt

def build_s2_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
    prompt = ""
    
    if examples:
        prompt += "--- EJEMPLOS ---\n"
        for ex in examples:
            text = ex.get('text', [])
            if isinstance(text, list):
                text = " ".join(text)
                
            prompt += f"Caso clínico:\n{text}\n"
            
            expected_json = {
                "premises": ex.get('premises', []),
                "claims": ex.get('claims', [])
            }
            prompt += f"Salida esperada:\n{json.dumps(expected_json, ensure_ascii=False, indent=2)}\n\n"
        prompt += "--- FIN DE EJEMPLOS ---\n\n"
        
    text = case.get('text', [])
    if isinstance(text, list):
        text = " ".join(text)
        
    prompt += f"Caso clínico a analizar:\n{text}\n\n"
    prompt += "Genera la salida estructurada en formato JSON estricto:"
    
    return prompt

def build_s3_prompt(relation: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
    prompt = ""
    
    if examples:
        prompt += "--- EJEMPLOS ---\n"
        for ex in examples:
            prompt += f"Premisa: \"{ex.get('head', '')}\"\n"
            prompt += f"Claim: \"{ex.get('tail', '')}\"\n"
            prompt += f"Salida esperada:\n{{\n  \"label\": \"{ex.get('label', '')}\"\n}}\n\n"
        prompt += "--- FIN DE EJEMPLOS ---\n\n"
        
    prompt += "Determina la relación para el siguiente par:\n"
    prompt += f"Premisa: \"{relation.get('head', '')}\"\n"
    prompt += f"Claim: \"{relation.get('tail', '')}\"\n\n"
    prompt += "Salida esperada (solo JSON):"
    
    return prompt