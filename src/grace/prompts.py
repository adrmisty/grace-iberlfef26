# prompts.py
# ----------------------------------------------------------
# configurations for prompts
# allows multilingual prompting
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
from typing import Dict, Any, List, Optional
from ..config import LANG
# --- static prompt builders -------------------------------------------------------------------------

SYSTEM_PROMPTS: Dict[str, Dict[str, str]] = {
    "es": {
        "SUBTASK_1": "Eres un experto clínico. Tu tarea es evaluar una lista numerada de oraciones de un caso clínico. Debes determinar si cada oración contiene evidencia médica RELEVANTE (síntomas, historial, pruebas) o si es irrelevante (texto de relleno, saludos, la pregunta final).\nREGLA ESTRICTA: Devuelve ÚNICAMENTE un objeto JSON donde las claves son los índices de las oraciones y los valores son booleanos (true/false).",
        "SUBTASK_2": "Eres un experto clínico especializado en NLP. Tu tarea es extraer fragmentos exactos de texto (spans). REGLAS CLAVE:\n1. PREMISAS: Hechos descriptivos del paciente (síntomas, antecedentes, exploración física). Extrae la frase exacta del texto sin alterar una sola coma.\n2. AFIRMACIONES (CLAIMS): TODAS las opciones de respuesta múltiple al final del caso. Cada opción es un claim distinto. Debes extraer su ID (1, 2, 3, etc.) y su texto exacto.\n3. EXCLUSIONES: NO extraigas la pregunta en sí (ej. '¿Qué diagnóstico...?').\nREGLA ESTRICTA: Devuelve ÚNICAMENTE un objeto JSON válido con las claves 'premises' (lista de strings) y 'claims' (lista de objetos con 'id' e 'text').",
        "SUBTASK_3": "Eres un razonador clínico. Se te dará una PREMISA (un hecho del paciente) y un CLAIM (una posible respuesta/diagnóstico). Tu tarea es determinar la relación argumentativa entre ellos:\n- 'Support': La premisa apoya, confirma o es consistente con el claim.\n- 'Attack': La premisa contradice, descarta o hace improbable el claim.\nREGLA ESTRICTA: Devuelve ÚNICAMENTE un JSON válido con la clave 'label'."
    },
    "en": {
        "SUBTASK_1": "You are a clinical expert. Your task is to evaluate a numbered list of sentences from a clinical case. You must determine if each sentence contains RELEVANT medical evidence (symptoms, medical history, tests) or if it is irrelevant (filler text, greetings, the final question).\nSTRICT RULE: Return ONLY a JSON object where the keys are the sentence indices and the values are booleans (true/false).",
        "SUBTASK_2": "You are a clinical expert specializing in NLP. Your task is to extract exact text spans. KEY RULES:\n1. PREMISES: Descriptive facts about the patient (symptoms, history, physical examination). Extract the exact phrase from the text without altering a single comma.\n2. CLAIMS: ALL multiple-choice answer options at the end of the case. Each option is a distinct claim. You must extract its ID (1, 2, 3, etc.) and its exact text.\n3. EXCLUSIONS: DO NOT extract the question itself (e.g., 'What is the most likely diagnosis?').\nSTRICT RULE: Return ONLY a valid JSON object with the keys 'premises' (list of strings) and 'claims' (list of objects with 'id' and 'text').",
        "SUBTASK_3": "You are a clinical reasoner. You will be given a PREMISE (a patient fact) and a CLAIM (a possible answer/diagnosis). Your task is to determine the argumentative relation between them:\n- 'Support': The premise supports, confirms, or is consistent with the claim.\n- 'Attack': The premise contradicts, rules out, or makes the claim unlikely.\nSTRICT RULE: Return ONLY a valid JSON object with the key 'label'."
    },
    "it": {
        "SUBTASK_1": "Sei un esperto clinico. Il tuo compito è valutare un elenco numerato di frasi tratte da un caso clinico. Devi determinare se ogni frase contiene prove mediche RILEVANTI (sintomi, anamnesi, esami) o se è irrilevante (testo riempitivo, saluti, la domanda finale).\nREGOLA RIGIDA: Restituisci SOLO un oggetto JSON in cui le chiavi sono gli indici delle frasi e i valori sono booleani (true/false).",
        "SUBTASK_2": "Sei un esperto clinico specializzato in NLP. Il tuo compito è estrarre porzioni di testo esatte (span). REGOLE CHIAVE:\n1. PREMESSE: Fatti descrittivi del paziente (sintomi, anamnesi, esame obiettivo). Estrai la frase esatta dal testo senza alterare una singola virgola.\n2. AFFERMAZIONI (CLAIMS): TUTTE le opzioni di risposta multipla alla fine del caso. Ogni opzione è un claim distinto. Devi estrarre il suo ID (1, 2, 3, ecc.) e il suo testo esatto.\n3. ESCLUSIONI: NON estrarre la domanda stessa (es. 'Qual è la diagnosi più probabile?').\nREGOLA RIGIDA: Restituisci SOLO un oggetto JSON valido con le chiavi 'premises' (elenco di stringhe) e 'claims' (elenco di oggetti con 'id' e 'text').",
        "SUBTASK_3": "Sei un ragionatore clinico. Ti verrà fornita una PREMESSA (un fatto del paziente) e un'AFFERMAZIONE (una possibile risposta/diagnosi). Il tuo compito è determinare la relazione argomentativa tra loro:\n- 'Support': La premessa supporta, conferma o è coerente con l'affermazione.\n- 'Attack': La premessa contraddice, esclude o rende improbabile l'affermazione.\nREGOLA RIGIDA: Restituisci SOLO un oggetto JSON valido con la chiave 'label'."
    },
    "fr": {
        "SUBTASK_1": "Vous êtes un expert clinique. Votre tâche consiste à évaluer une liste numérotée de phrases d'un cas clinique. Vous devez déterminer si chaque phrase contient des preuves médicales PERTINENTES (symptômes, antécédents, examens) ou si elle n'est pas pertinente (texte de remplissage, salutations, la question finale).\nRÈGLE STRICTE : Renvoyez UNIQUEMENT un objet JSON où les clés sont les index des phrases et les valeurs sont des booléens (true/false).",
        "SUBTASK_2": "Vous êtes un expert clinique spécialisé en NLP. Votre tâche consiste à extraire des extraits de texte exacts (spans). RÈGLES CLÉS :\n1. PRÉMISSES : Faits descriptifs sur le patient (symptômes, antécédents, examen physique). Extrayez la phrase exacte du texte sans modifier une seule virgule.\n2. AFFIRMATIONS (CLAIMS) : TOUTES les options de réponse à choix multiples à la fin du cas. Chaque option est une affirmation distincte. Vous devez extraire son ID (1, 2, 3, etc.) et son texte exact.\n3. EXCLUSIONS : N'extrayez PAS la question elle-même (par ex., 'Quel est le diagnostic le plus probable ?').\nRÈGLE STRICTE : Renvoyez UNIQUEMENT un objet JSON valide avec les clés 'premises' (liste de chaînes) et 'claims' (liste d'objets avec 'id' et 'text').",
        "SUBTASK_3": "Vous êtes un raisonneur clinique. On vous donnera une PRÉMISSE (un fait concernant le patient) et une AFFIRMATION (une réponse/diagnostic possible). Votre tâche est de déterminer la relation argumentative entre elles :\n- 'Support' : La prémisse soutient, confirme ou est cohérente avec l'affirmation.\n- 'Attack' : La prémisse contredit, exclut ou rend l'affirmation improbable.\nRÈGLE STRICTE : Renvoyez UNIQUEMENT un objet JSON valide avec la clé 'label'."
    }
}

SYSTEM_PROMPTS_alvaro_alex: Dict[str, Dict[str, str]] = {
    "es": {
        "SUBTASK_1": "Eres un experto clínico. Tu tarea es evaluar una lista numerada de oraciones de un caso clínico. Debes determinar si cada oración contiene evidencia médica RELEVANTE (síntomas, historial, pruebas) o si es irrelevante (texto de relleno, saludos, la pregunta final).\nREGLA ESTRICTA: Devuelve ÚNICAMENTE un objeto JSON donde las claves son los índices de las oraciones y los valores son booleanos (true/false).",
        "SUBTASK_2": "Eres un experto clínico especializado en NLP. Tu tarea es extraer fragmentos exactos de texto (spans). REGLAS CLAVE:\n1. PREMISAS: Hechos descriptivos del paciente (síntomas, antecedentes, exploración física). Extrae la frase exacta del texto sin alterar una sola coma.\n2. AFIRMACIONES (CLAIMS): TODAS las opciones de respuesta múltiple al final del caso. Cada opción es un claim distinto. Debes extraer su ID (1, 2, 3, etc.) y su texto exacto.\n3. EXCLUSIONES: NO extraigas la pregunta en sí (ej. '¿Qué diagnóstico...?').\nREGLA ESTRICTA: Devuelve ÚNICAMENTE un objeto JSON válido con las claves 'premises' (lista de strings) y 'claims' (lista de objetos con 'id' e 'text').",
        "SUBTASK_3": "Eres un razonador clínico. Se te dará una PREMISA (un hecho del paciente) y un CLAIM (una posible respuesta/diagnóstico). Tu tarea es determinar la relación argumentativa entre ellos:\n- 'Support': La premisa apoya, confirma o es consistente con el claim.\n- 'Attack': La premisa contradice, descarta o hace improbable el claim.\nREGLA ESTRICTA: Devuelve ÚNICAMENTE un JSON válido con la clave 'label'."
    },
}


EX_STRINGS = {
    "es": {"ex_start": "--- EJEMPLOS ---", "ex_end": "--- FIN DE EJEMPLOS ---", "case": "Caso clínico:", "sentences": "Oraciones:", "expected": "Salida esperada:", "analyze": "Caso clínico a analizar:", "premise": "Premisa:", "generate": "Genera el JSON de salida:"},
    "en": {"ex_start": "--- EXAMPLES ---", "ex_end": "--- END OF EXAMPLES ---", "case": "Clinical case:", "sentences": "Sentences:", "expected": "Expected output:", "analyze": "Clinical case to analyze:", "premise": "Premise:", "generate": "Generate the output JSON:"},
    "it": {"ex_start": "--- ESEMPI ---", "ex_end": "--- FINE DEGLI ESEMPI ---", "case": "Caso clinico:", "sentences": "Frasi:", "expected": "Uscita attesa:", "analyze": "Caso clinico da analizzare:", "premise": "Premessa:", "generate": "Genera il JSON di output:"},
    "fr": {"ex_start": "--- EXEMPLES ---", "ex_end": "--- FIN DES EXEMPLES ---", "case": "Cas clinique :", "sentences": "Phrases :", "expected": "Sortie attendue :", "analyze": "Cas clinique à analyser :", "premise": "Prémisse :", "generate": "Générez le JSON de sortie :"}
}

# --- dynamic prompt builders ---

def build_s1_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = LANG) -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS[lang])
    prompt = ""
    
    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            prompt += f"{ui['sentences']}\n"
            for i, sent in enumerate(ex.get('text', [])):
                prompt += f"[{i}] {sent}\n"
            prompt += f"{ui['expected']}\n\{{\n"
            labels = ex.get('relevance_labels', {})
            items = [f'  "{k}": {"true" if v else "false"}' for k, v in labels.items()]
            prompt += ",\n".join(items) + "\n}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
        
    prompt += f"{ui['analyze']}\n{ui['sentences']}\n"
    for i, sent in enumerate(case.get('text', [])):
        prompt += f"[{i}] {sent}\n"
        
    prompt += f"\n{ui['generate']}"
    return prompt

def build_s2_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = LANG) -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS[lang])
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

def build_s3_prompt(relation: Dict[str, Any], examples: Optional[List[Dict[str, Any]]], lang: str = LANG) -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS[lang])
    prompt = ""
    
    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            prompt += f"{ui['premise']} \"{ex.get('head', '')}\"\n"
            prompt += f"Claim: \"{ex.get('tail', '')}\"\n"
            prompt += f"{ui['expected']}\n{{\n  \"label\": \"{ex.get('label', '')}\"\n}}\n\n"
        prompt += f"{ui['ex_end']}\n\n"
        
    prompt += f"{ui['premise']} \"{relation.get('head', '')}\"\n"
    prompt += f"Claim: \"{relation.get('tail', '')}\"\n\n"
    prompt += f"{ui['expected']} (JSON):"
    return prompt