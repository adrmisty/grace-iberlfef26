# prompts.py
# ----------------------------------------------------------
# configurations for prompts
# allows multilingual prompting
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
from typing import Dict, Any, List, Optional

# --- static prompt builders -------------------------------------------------------------------------

from typing import Dict

SYSTEM_PROMPTS: Dict[str, Dict[str, str]] = {
    "es": {
        "SUBTASK_1": (
            "Eres un experto clínico. Tu tarea es evaluar una lista numerada de oraciones de un caso clínico. "
            "Debes determinar si cada oración contiene evidencia médica RELEVANTE (síntomas, historial, pruebas) "
            "o si es IRRELEVANTE (texto de relleno, saludos, la pregunta final).\n\n"
            "Restricciones obligatorias:\n"
            "- \"sentence_relevancy\" debe tener exactamente una etiqueta por cada oración recibida, y usa solo \"relevant\" o \"not-relevant\".\n"
            "- Devuelve únicamente JSON válido.\n\n"
            "Formato obligatorio de salida:\n"
            "{\n"
            "  \"sentence_relevancy\": [\n"
            "    \"relevant\",\n"
            "    \"not-relevant\"\n"
            "  ]\n"
            "}"
        ),
        "SUBTASK_2": (
            "Eres un experto clínico especializado en NLP. Tu tarea es extraer fragmentos exactos de texto (spans). REGLAS CLAVE:\n"
            "1. PREMISAS: Hechos descriptivos del paciente (síntomas, antecedentes, exploración física). Extrae la frase exacta del texto sin alterar una sola coma.\n"
            "2. AFIRMACIONES (CLAIMS): TODAS las opciones de respuesta múltiple al final del caso. Cada opción es un claim distinto. Debes extraer su ID (1, 2, 3, etc.) y su texto exacto.\n"
            "3. EXCLUSIONES: NO extraigas la pregunta en sí (ej. '¿Qué diagnóstico...?').\n\n"
            "Restricciones obligatorias para las Premises:\n"
            "- Cada \"source_index\" debe ser el índice de la oración de la que sale la Premise.\n"
            "- Cada \"text\" debe aparecer literalmente dentro de la oración indicada.\n"
            "- Cada \"text\" debe ser el menor fragmento clínicamente suficiente, no la oración completa.\n"
            "- Devuelve únicamente JSON válido.\n\n"
            "Formato obligatorio de salida:\n"
            "{"
            "  \"premises\": ["
            "    {"
            "      \"local_id\": \"p1\","
            "      \"source_index\": 0,"
            "      \"text\": \"fragmento exacto mínimo\""
            "    }\n"
            "  ],\n"
            "  \"claims\": [\n"
            "    {"
            "      \"id\": \"1\","
            "      \"text\": \"texto exacto de la opción\""
            "    }"
            "  ]"
            "}"
        ),
        "SUBTASK_3": (
            "Eres un razonador clínico. Se te dará una PREMISA (un hecho del paciente) y un CLAIM (una posible respuesta/diagnóstico). "
            "Tu tarea es determinar la relación argumentativa entre ellos:\n"
            "- 'Support': La premisa apoya, confirma o es consistente con el claim.\n"
            "- 'Attack': La premisa contradice, descarta o hace improbable el claim.\n\n"
            "Restricciones obligatorias:\n"
            "- Cada \"premise_id\" debe ser el ID proporcionado para la Premise analizada.\n"
            "- Cada \"claim_id\" debe corresponder a la opción recibida.\n"
            "- Usa solo \"Support\" o \"Attack\".\n"
            "- Devuelve únicamente JSON válido.\n\n"
            "Formato obligatorio de salida:\n"
            "{"
            "  \"relations\": ["
            "    {"
            "      \"premise_id\": \"p1\","
            "      \"claim_id\": \"3\","
            "      \"relation_type\": \"Support\""
            "    }"
            "  ]"
            "}"
        )
    }
}

EX_STRINGS = {
    "es": {"ex_start": "--- EJEMPLOS ---", "ex_end": "--- FIN DE EJEMPLOS ---", "case": "Caso clínico:", "sentences": "Oraciones:", "expected": "Salida esperada:", "analyze": "Caso clínico a analizar:", "premise": "Premisa:", "generate": "Genera el JSON de salida:"},
}

def build_s1_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = "es") -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS["es"])
    prompt = ""
    
    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            prompt += f"{ui['sentences']}\n"
            num_sentences = len(ex.get('text', []))
            for i, sent in enumerate(ex.get('text', [])):
                prompt += f"[{i}] {sent}\n"
            
            # ** expected format: list of [relevant, not-relevant...]
            labels = ex.get('relevance_labels', {})
            relevancy_list = ["relevant" if labels.get(str(i), labels.get(i, False)) else "not-relevant" for i in range(num_sentences)]
            expected_json = {"sentence_relevancy": relevancy_list}
            
            prompt += f"{ui['expected']}\n{json.dumps(expected_json, ensure_ascii=False, indent=2)}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
        
    prompt += f"{ui['analyze']}\n{ui['sentences']}\n"
    for i, sent in enumerate(case.get('text', [])):
        prompt += f"[{i}] {sent}\n"
        
    prompt += f"\n{ui['generate']}"
    return prompt


def build_s2_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = "es") -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS["es"])
    prompt = ""

    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            text = ex.get('text', [])
            if isinstance(text, list): text = " ".join(text)
            prompt += f"{ui['case']}\n{text}\n"
            expected_json = {"premises": ex.get('premises', []), "claims": ex.get('claims', [])}
            prompt += f"{ui['expected']}\n{json.dumps(expected_json, ensure_ascii=False, indent=2)}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
        
    text = case.get('text', [])
    if isinstance(text, list): text = " ".join(text)
        
    prompt += f"{ui['analyze']}\n{text}\n\n"
    prompt += f"{ui['generate']}"
    return prompt


def build_s3_prompt(relation: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = "es") -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS["es"])
    prompt = ""

    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            prompt += f"{ui['premise']} \"{ex.get('head', '')}\"\n"
            prompt += f"Claim: \"{ex.get('tail', '')}\"\n"
            prompt += f"{ui['expected']}\n{{\n  \"label\": \"{ex.get('label', '')}\"\n}}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
    
    case_text = relation.get('text', [])
    if isinstance(case_text, list): case_text = " ".join(case_text)
    
    prompt += f"{ui['case']}\n{case_text}\n\n"
    prompt += f"{ui['premise']} \"{relation.get('head', '')}\"\n"
    prompt += f"Claim: \"{relation.get('tail', '')}\"\n\n"
    prompt += f"{ui['expected']} (JSON):"
    return prompt


# --- álvaro y alex

SYSTEM_PROMPTS_AA: Dict[str, Dict[str, str]] = {
    "es": {
        "SUBTASK_1": (
            "Eres un experto médico. Tu tarea es la Detección de Oraciones de Evidencia.\n"
            "Analiza la siguiente oración dentro del caso clínico y determina si es \"relevant\" o \"not-relevant\" para apoyar o refutar las opciones de respuesta.\n\n"
            "Caso Clínico:\n{AQUI SE INSERTA EL CONTEXTO}\n\n"
            "Opciones:\n{AQUI SE INSERTAN LAS OPCIONES}\n\n"
            "Oración a evaluar:\n{AQUÍ SE INSERTA LA ORACIÓN A EVALUAR}\n\n"
            "Responde únicamente con \"relevant\" o \"not-relevant\"."
        ),
        "SUBTASK_2": (
            "Eres un experto en razonamiento clínico y extracción de información. Tu tarea es identificar y "
            "extraer fragmentos de texto exactos que representen 'Premises' o 'Claims' dentro de una oración "
            "específica, utilizando el caso clínico completo solo como contexto de fondo.\n\n"
            "Definiciones:\n"
            "- Premise: Evidencia clínica objetiva (hechos, mediciones, síntomas, observaciones o antecedentes médicos del paciente).\n"
            "- Claim: Opciones de respuesta o hipótesis (diagnósticos candidatos, propuestas de tratamiento o pronósticos).\n\n"
            "Reglas de extracción:\n"
            "1. Evalúa ÚNICAMENTE la \"Oración a analizar\".\n"
            "2. El fragmento extraído debe ser una copia EXACTA (respetando mayúsculas, puntuación y espacios) de cómo aparece en la oración. Una 'Claim' puede abarcar la oración completa.\n"
            "3. Si la oración no contiene ninguna 'Premise' ni 'Claim' (ej. texto de relleno o preguntas genéricas), debes devolver un array vacío: []\n"
            "4. Responde ÚNICAMENTE con un array JSON válido, sin introducciones ni explicaciones previas.\n"
            "5. Formato de salida: [{{\"text\": fragmento de texto, \"type\": Premise/Claim}}]\n"
            "Contexto:\n{AQUI SE INSERTA EL CONTEXTO}\n"
            "Oración a analizar:\n{AQUÍ SE INSERTA LA ORACIÓN A ANALIZAR}\n\n"
        ),
        "SUBTASK_3": (
            "Eres un experto clínico. Tu tarea es evaluar la relación argumentativa entre una evidencia (Premise) y una opción candidata (Claim) basándote en el caso clínico proporcionado.\n\n"
            "Caso Clínico:\n{AQUI SE INSERTA EL CONTEXTO}\n\n"
            "Evidencia (Premise):\n{AQUI SE INSERTA LA PREMISE}\n\n"
            "Opción (Claim):\n{AQUI SE INSERTA LA CLAIM}\n\n"
            "Las posibles relaciones entre 'premise' y 'claim' son:\n- Support: Si la premise apoya, confirma o es consistente con la claim.\n- Attack: Si la premise contradice, refuta, descarta o hace improbable la claim.\n- Nothing: Si la premise y la claim no tienen relación.\n"
            "Responde únicamente con 'Support', 'Attack' o 'Nothing'."
        )
    }
}

