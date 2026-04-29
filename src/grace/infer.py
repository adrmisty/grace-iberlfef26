# global.py
# ----------------------------------------------------------
# configurations for one-step inference global prompt
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

import json
from typing import Dict, Any, List, Optional

# --- global prompt ---

GLOBAL_SYSTEM_PROMPT = """
Eres un experto médico en razonamiento clínico MIR y extracción de argumentos clínicos.
Tu tarea es resolver tres subtareas de razonamiento clínico en una única inferencia.

Recibirás:
1. Un caso clínico completo.
2. Una lista de oraciones del contexto clínico, cada una con un índice.
3. Una lista de opciones de respuesta, cada una con un id. A las cuales denominaremos Claims.

IMPORTANTE:
Tu tarea consiste en:
1. Clasificar cada oración (separadas por ".") del contexto como "relevant" o "not-relevant".
2. Extraer únicamente Premises mínimas desde las oraciones clasificadas como "relevant".
3. Relacionar las Premises extraídas con las Claims/opciones usando el id de la opción.

Subtarea 1 (Evidence Sentence Detection)
Clasifica cada oración del contexto clínico como:
- "relevant": contiene evidencia clínica útil para apoyar o refutar alguna opción (derivar un tratamiento o diagnóstico).
- "not-relevant": no aporta evidencia clínica útil para decidir entre las opciones/claims (derivar un tratamiento o diagnóstico).

Reglas para la Subtarea 1:
- Las oraciones con síntomas, signos, antecedentes, resultados de pruebas, evolución temporal, factores de riesgo, negaciones clínicas o hallazgos clínicos suelen ser "relevant".
- Las preguntas genéricas como:"¿Cuál es el diagnóstico más probable?", "¿Cuál es el tratamiento indicado?" o frases que solo introducen la pregunta, son "not-relevant".
- Una oración no debe ser "relevant" solo por introducir la pregunta.

Subtarea 2 (Minimal Premise Span Detection):
Extrae fragmentos exactos de texto que sean Premises.

Una Premise es una unidad mínima de evidencia clínica objetiva del caso relativo al paciente:
- síntomas
- signos
- antecedentes clínicos
- medicación que tome
- edad o sexo si son clínicamente relevantes
- duración o evolución temporal
- resultado de prueba
- hallazgo de exploración
- factor de riesgo
- ausencia o negación clínica relevante
- datos médicos que apoyen o refute una opción

Reglas estrictas para Premises:
1. Extrae Premises solo desde oraciones clasificadas como "relevant".
2. El texto debe ser una copia EXACTA de un fragmento de la oración original.
3. No reformules. No inventes.
5. No incluyas offsets.
6. No extraigas texto de las opciones como Premise.
7. No extraigas la pregunta final como Premise.
8. No devuelvas una oración completa si contiene varias unidades clínicas.
9. Si una oración contiene varias evidencias clínicas, divide la oración en varias Premises mínimas.
10. Cada Premise debe expresar una sola unidad o idea clínica.
11. Prefiere el span más corto que conserve el significado clínico, sin reformular o inventar estos sub-span.
13. Mantén modificadores clínicamente importantes, por ejemplo duración, localización, severidad o resultado de prueba.
14. No incluyas introducciones, conectores ni relleno, solo la evidencia clínica concreta.

Ejemplos de buenas Premises mínimas:
- "crisis de pánico durante sus actuaciones públicas"
- "intenso miedo a quedar bloqueado"
- "sería humillante"
- "En el resto de sus actividades diarias no experimenta este temor"
- "tres abortos espontáneos"
- "No ha tenido ningún embarazo a término"
- "hipertensión arterial"
- "dolor hipogástrico de inicio brusco"
- "metrorragia escasa de sangre oscura"
- "sangre oscura"
- "frecuencia cardiaca fetal no reactivo"
- "gorra de trabajo porque precisa cada vez tallas mayores"
- "fosfatasa alcalina con niveles séricos cuatro veces mayor de lo normal"

Ejemplos de malas Premises:
- Una oración completa con varios datos mezclados.
- Una frase introductoria sin contenido clínico específico.
- La pregunta diagnóstica final.
- Una opción de respuesta.
- Texto reformulado.
- Un resumen en vez de una copia exacta.

Subtarea 3 (Argumentative Relation Detection):
Relaciona cada Premise con una o varias Claims cuando exista una relación clara.

Las Claims son las opciones de respuesta recibidas.
Debes referirte a ellas usando su id.

Tipos de relación:
- "Support": la Premise apoya, favorece, confirma o es consistente con esa opción.
- "Attack": la Premise contradice, descarta, debilita o hace improbable esa opción.

No incluyas relaciones dudosas, o falta de relación.

Formato obligatorio de salida:

{
  "sentence_relevancy": [
    "relevant",
    "not-relevant"
  ],
  "premises": [
    {
      "local_id": "p1",
      "source_index": 0,
      "text": "fragmento exacto mínimo"
    }
  ],
  "relations": [
    {
      "premise_id": "p1",
      "claim_id": "3",
      "relation_type": "Support"
    }
  ]
}

Restricciones obligatorias:
- "sentence_relevancy" debe tener exactamente una etiqueta por cada oración recibida, y usa solo "relevant" o "not-relevant".
- Cada "source_index" debe ser el índice de la oración de la que sale la Premise.
- Cada "text" debe aparecer literalmente dentro de la oración indicada.
- Cada "text" debe ser el menor fragmento clínicamente suficiente, no la oración completa.
- Cada "premise_id" debe existir en "premises".
- Cada "claim_id" debe corresponder a una opción recibida.
- Usa solo "Support" o "Attack".
- Devuelve únicamente JSON válido.
""".strip()


EX_STRINGS = {
    "es": {"ex_start": "--- EJEMPLOS ---", "ex_end": "--- FIN DE EJEMPLOS ---", "case": "Caso clínico:", "sentences": "Oraciones:", "options": "Opciones (Claims):", "expected": "Salida esperada:", "analyze": "Caso clínico a analizar:", "premise": "Premisa:", "generate": "Genera el JSON de salida:"},
}

# --- dynamic prompt builders ---


def build_usr_global_prompt(case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]] = None, example_relations: Optional[List[Dict[str, Any]]] = None, lang: str = "es") -> str:
    ui = EX_STRINGS.get(lang, EX_STRINGS["es"])
    prompt = ""

    if examples:
        prompt += f"{ui['ex_start']}\n"
        for ex in examples:
            prompt += f"{ui['case']}\n"
            text = ex.get('text', [])
            if isinstance(text, list): text = " ".join(text)
            prompt += f"{text}\n\n"
            
            prompt += f"{ui['sentences']}\n"
            for i, sent in enumerate(ex.get('text', [])):
                prompt += f"[{i}] {sent}\n"
                
            prompt += f"\n{ui['options']}\n"
            claims = ex.get('claims', [])
            for c in claims:
                prompt += f"- id: {c.get('id', '')}, text: \"{c.get('text', '')}\"\n"

            # -----------------------
            # GLOBAL TARGET EXAMPLES
            # -----------------------
            global_target = {
                "sentence_relevancy": [],
                "premises": [],
                "relations": []
            }
            
            # ** S1: sentence relevancy **
            labels = ex.get('relevance_labels', {})
            for i in range(len(ex.get('text', []))):
                is_rel = labels.get(str(i), labels.get(i, False))
                global_target["sentence_relevancy"].append("relevant" if is_rel else "not-relevant")
                
            # ** S2: premises/claims + their offset **
            premises = ex.get('premises', [])
            premise_text_to_id = {}
            for idx, p_text in enumerate(premises):
                p_id = f"p{idx+1}"
                premise_text_to_id[p_text] = p_id
                
                s_idx = 0
                for i, sent in enumerate(ex.get('text', [])):
                    if p_text in sent:
                        s_idx = i
                        break
                        
                global_target["premises"].append({
                    "local_id": p_id,
                    "source_index": s_idx,
                    "text": p_text
                })
            
            # ** S3: relations **
            claim_text_to_id = {c.get('text', ''): str(c.get('id', '')) for c in claims}
            case_rels = []
            if example_relations:
                case_rels = [r for r in example_relations if r.get("case_id") == ex.get("id")]
            else:
                case_rels = ex.get("annotations", {}).get("relations", [])
                
            for r in case_rels:
                # unified/CasiMedicos (head/tail) o GRACE estandar
                p_text = r.get("head")
                c_text = r.get("tail")
                label = r.get("label", r.get("relation_type", "")).capitalize()
                
                if p_text and c_text:
                    p_id = premise_text_to_id.get(p_text)
                    c_id = claim_text_to_id.get(c_text)
                    
                    if p_id and c_id and label in ["Support", "Attack"]:
                        global_target["relations"].append({
                            "premise_id": p_id,
                            "claim_id": c_id,
                            "relation_type": label
                        })

            prompt += f"\n{ui['expected']}\n{json.dumps(global_target, ensure_ascii=False, indent=2)}\n\n"
        prompt += f"{ui['ex_end']}\n\n"

    # ** current case **
    case_text = case.get('text', [])
    if isinstance(case_text, list): case_text = " ".join(case_text)
    
    prompt += f"{ui['analyze']}\n{ui['case']}\n{case_text}\n\n"
    
    prompt += f"{ui['sentences']}\n"
    for i, sent in enumerate(case.get('text', [])):
        prompt += f"[{i}] {sent}\n"
        
    prompt += f"\n{ui['options']}\n"
    for c in case.get('claims', []):
        prompt += f"- id: {c.get('id', '')}, text: \"{c.get('text', '')}\"\n"

    prompt += f"\n{ui['generate']}"
    return prompt
