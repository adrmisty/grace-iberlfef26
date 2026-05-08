# prompts.py
# ----------------------------------------------------------
# configurations for prompts split per subtasks
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# may-2026

import json
from typing import Dict, Any, List, Optional

# --- static prompt builders -------------------------------------------------------------------------

from typing import Dict

SYSTEM_PROMPTS_v0: Dict[str, Dict[str, str]] = {
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
        
        # ** fix: include 'None' as valid relation **
        "SUBTASK_3": (
            "Eres un razonador clínico. Se te dará una PREMISA (un hecho del paciente) y un CLAIM (una posible respuesta/diagnóstico). "
            "Tu tarea es determinar la relación argumentativa entre ellos:\n"
            "- 'Support': La premisa apoya, confirma o es consistente con el claim.\n"
            "- 'Attack': La premisa contradice, descarta o hace improbable el claim.\n"
            "- 'None': La premisa no tiene una relación argumentativa directa o clínicamente relevante con el claim.\n\n"
            "Restricciones obligatorias:\n"
            "- Cada \"premise_id\" debe ser el ID proporcionado para la Premise analizada.\n"
            "- Cada \"claim_id\" debe corresponder a la opción recibida.\n"
            "- Usa solo \"Support\", \"Attack\" o \"None\".\n"
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

# --- prompts de Álvaro y Álex ---

SYSTEM_PROMPTS_v1: Dict[str, Dict[str, str]] = {
    "es": {
        "SUBTASK_1": (
            "Eres un experto médico. Tu tarea es la Detección de Oraciones de Evidencia.\n"
            "Analiza la siguiente lista numerada de oraciones de un caso clínico y determina si cada una es \"relevant\" o \"not-relevant\" para apoyar o refutar diagnósticos/tratamientos.\n\n"
            "Formato obligatorio de salida:\n"
            "{\n"
            "  \"sentence_relevancy\": [\n"
            "    \"relevant\",\n"
            "    \"not-relevant\"\n"
            "  ]\n"
            "}"
        ),
        "SUBTASK_2": (
            "Eres un experto en razonamiento clínico y extracción de información. Tu tarea es identificar y "
            "extraer fragmentos de texto exactos que representen 'Premises' o 'Claims' dentro del caso clínico proporcionado.\n\n"
            "Definiciones:\n"
            "- Premise: Evidencia clínica objetiva (hechos, mediciones, síntomas, observaciones).\n"
            "- Claim: Opciones de respuesta o hipótesis.\n\n"
            "Reglas de extracción:\n"
            "1. El fragmento extraído debe ser una copia EXACTA del texto original.\n"
            "2. Asigna a cada Premise un 'local_id' correlativo (p1, p2...) y el 'source_index' de la oración donde aparece.\n"
            "3. Extrae las Claims con su respectivo ID.\n\n"
            "Formato obligatorio de salida:\n"
            "{\n"
            "  \"premises\": [\n"
            "    {\n"
            "      \"local_id\": \"p1\",\n"
            "      \"source_index\": 0,\n"
            "      \"text\": \"fragmento exacto mínimo\"\n"
            "    }\n"
            "  ],\n"
            "  \"claims\": [\n"
            "    {\n"
            "      \"id\": \"1\",\n"
            "      \"text\": \"texto exacto de la opción\"\n"
            "    }\n"
            "  ]\n"
            "}"
        ),
        "SUBTASK_3": (
            "Eres un experto clínico. Tu tarea es evaluar la relación argumentativa entre una evidencia (Premise) y una opción candidata (Claim) basándote en el caso clínico proporcionado.\n\n"
            "Las posibles relaciones son:\n"
            "- Support: Si la premise apoya, confirma o es consistente con la claim.\n"
            "- Attack: Si la premise contradice, refuta, descarta o hace improbable la claim.\n\n"
            "Devuelve la relación utilizando los IDs proporcionados en la entrada.\n\n"
            "Formato obligatorio de salida:\n"
            "{\n"
            "  \"relations\": [\n"
            "    {\n"
            "      \"premise_id\": \"p1\",\n"
            "      \"claim_id\": \"c1\",\n"
            "      \"relation_type\": \"Support\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )
    }
}

EX_STRINGS = {
    "es": {
        "ex_start": "Ejemplos:", 
        "ex_end": "Fin de ejemplos.", 
        "case": "Caso clínico:", 
        "sentences": "Oraciones:", 
        #"options": "Opciones:",
        "expected": "Salida esperada:", 
        "analyze": "Caso clínico a analizar:", 
        "premise": "Premisa:", 
        "generate": "Genera el JSON de salida:"
    },
}


def build_s1_usr_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = "es") -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS["es"])
    prompt = ""
    
    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            prompt += f"{ui['sentences']}\n"
            num_sentences = len(ex.get('text', []))
            for i, sent in enumerate(ex.get('text', [])):
                prompt += f"[{i}] {sent}\n"
            
            labels = ex.get('relevance_labels', {})
            # ** format examples itno expected S1 format (list of [not-]relevant)
            relevancy_list = ["relevant" if labels.get(str(i), labels.get(i, False)) else "not-relevant" for i in range(num_sentences)]
            expected_json = {"sentence_relevancy": relevancy_list}
            
            prompt += f"{ui['expected']}\n{json.dumps(expected_json, ensure_ascii=False, indent=2)}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
        
    prompt += f"{ui['analyze']}\n{ui['sentences']}\n"
    for i, sent in enumerate(case.get('text', [])):
        prompt += f"[{i}] {sent}\n"
        
    prompt += f"\n{ui['generate']}"
    return prompt


def build_s2_usr_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = "es") -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS["es"])
    prompt = ""

    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            text = ex.get('text', [])
            if isinstance(text, list): text = " ".join(text)
            prompt += f"{ui['case']}\n{text}\n"
            
            # ** format examples into expected S2 format (dict of local_id, source_index and entity text)**
            structured_premises = []
            for p_idx, p_text in enumerate(ex.get('premises', [])):
                s_idx = 0
                for i, sent in enumerate(ex.get('text', [])):
                    if p_text in sent:
                        s_idx = i
                        break
                structured_premises.append({
                    "local_id": f"p{p_idx+1}",
                    "source_index": s_idx,
                    "text": p_text
                })
                
            expected_json = {"premises": structured_premises, "claims": ex.get('claims', [])}
            prompt += f"{ui['expected']}\n{json.dumps(expected_json, ensure_ascii=False, indent=2)}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
        
    text = case.get('text', [])
    if isinstance(text, list): text = " ".join(text)
        
    prompt += f"{ui['analyze']}\n{text}\n\n"
    prompt += f"{ui['generate']}"
    return prompt


def build_s3_usr_prompt(relation: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = "es") -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS["es"])
    prompt = ""

    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            prompt += f"Premise [p1]: \"{ex.get('head', '')}\"\n"
            prompt += f"Claim [c1]: \"{ex.get('tail', '')}\"\n"
            
            # ** format examples into expected S3 format (dictionary of entity id and relation)**
            label = str(ex.get('label', '')).capitalize()
            expected_json = {
                "relations": [{"premise_id": "p1", "claim_id": "c1", "relation_type": label}]
            }
            prompt += f"{ui['expected']}\n{json.dumps(expected_json, ensure_ascii=False, indent=2)}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
    
    case_text = relation.get('text', [])
    if isinstance(case_text, list): case_text = " ".join(case_text)
    
    prompt += f"{ui['case']}\n{case_text}\n\n"
    prompt += f"Premise [p1]: \"{relation.get('head', '')}\"\n"
    prompt += f"Claim [c1]: \"{relation.get('tail', '')}\"\n\n"
    prompt += f"{ui['expected']} (JSON):"
    return prompt