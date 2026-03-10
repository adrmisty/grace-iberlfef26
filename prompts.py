# prompts.py
# ----------------------------------------------------------
# configurations for prompts
# ----------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

from typing import Dict

SYSTEM: Dict[str, str] = {
    "SUBTASK_1": (
            "Eres un asistente médico experto. Tu tarea es clasificar si una oración "
            "proporciona información clínica relevante (true) o no (false) para "
            "apoyar o refutar un diagnóstico o tratamiento propuesto."
        ),
    
    "SUBTASK_2": "",
    "SUBTASK_3": "",
}
