# model.py
# ----------------------------------------------------------------------------------------
# zero-shot & few-shot evaluation on Qwen3.5 for CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import json
import torch
import logging
import config as settings
import prompts
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

class GraceModel:
    def __init__(self, model_size: str = "2B", data_dir: str = settings.BASE_DATA_DIR):
        """
        Inits the Qwen3.5 evaluator, with model_size: "2B", "4B", or "8B"
        """
        self.model_id = f"Qwen/Qwen3.5-{model_size}"
        self.data_dir = Path(data_dir)
        self.splits_dir = self.data_dir / "splits"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()

    # --- model init -------------------------------------------------------------------------

    def _load_model(self):
        """Loads the model for zero/few-shot prompting."""
        logging.info(f"> Loading {self.model_id} on {self.device} for evaluation...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        logging.info(f"\t >>> {self.model_id} model for [zero-shot/few-shot] prompting loaded successfully!!!")

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, prefill: str = "") -> str:
        """Generates a response using Qwen's chat template, supporting assistant prefilling."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # ** prefill ** to guide model's output instead of verbose thinking process
        if prefill:
            text += prefill
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        decoded_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return prefill + decoded_text
    
    # --- subtasks prompting -------------------------------------------------------------------------

    def run_subtask_1(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        logging.info(f"> Subtask 1 (relevance detection)...")
        results = []
        
        for case in test_data:
            user_prompt = self._s1_fewshot(case, few_shot_examples)
            # prefill: start with JSON output
            response = self._generate(prompts.SYSTEM["SUBTASK_1"], user_prompt, prefill="{\n")
            results.append({"id": case.get("id"), "prediction": response})
            
        return results

    def run_subtask_2(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        logging.info(f"> Subtask 2 (span detection)...")
        results = []
        
        for case in test_data:
            user_prompt = self._s2_fewshot(case, few_shot_examples)
            # prefill: Premises:\n- to guide the model to start with the required output
            response = self._generate(prompts.SYSTEM["SUBTASK_2"], user_prompt, prefill="Premisas:\n-")
            results.append({"id": case.get("id"), "prediction": response})
            
        return results

    def run_subtask_3(self, test_relations: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None, max_new_tokens: int = 128):
        logging.info(f"> Subtask 3 (relation detection)...")
        results = []
        
        for relation in test_relations:
            user_prompt = self._build_s3_prompt(relation, few_shot_examples)
            # ++ TOKENS
            response = self._generate(prompts.SYSTEM["SUBTASK_3"], user_prompt, max_new_tokens=max_new_tokens)
            results.append({"id": relation.get("id"), "prediction": response.strip()})
            
        return results

    # --- prompt building -------------------------------------------------------------------------
    
    def _s1_fewshot(self, case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
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

    def _s2_fewshot(self, case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
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

    def _build_s3_prompt(self, relation: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
        prompt = "AVISO IMPORTANTE: Eres un script de automatización. Genera ÚNICAMENTE la salida estructurada solicitada. Prohibido generar 'Thinking Process' o explicaciones.\n\n"
        
        if examples:
            prompt += "--- EJEMPLOS ---\n"
            for ex in examples:
                prompt += f"Premise: \"{ex.get('head', '')}\"\n"
                prompt += f"Claim: \"{ex.get('tail', '')}\"\n"
                prompt += f"Relación: {ex.get('label', '')}\n\n"
            prompt += "--- FIN DE EJEMPLOS ---\n\n"
        else:
            prompt += "FORMATO ESPERADO: Debes responder ÚNICAMENTE con la palabra 'Support' o la palabra 'Attack'.\n\n"
            
        prompt += "Clasifica la siguiente relación:\n\n"
        prompt += f"Premise: \"{relation.get('head', '')}\"\n"
        prompt += f"Claim: \"{relation.get('tail', '')}\"\n"
        prompt += "Relación:"
        prompt += "\nREGLA ESTRICTA: Responde ÚNICAMENTE con la palabra 'Support' o 'Attack'. NO escribas 'Thinking Process' ni des explicaciones."
        return prompt