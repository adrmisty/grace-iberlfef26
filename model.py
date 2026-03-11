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
        inits the Qwen3.5 evaluator.
        :param model_size: "2B", "4B", or "8B"
        """
        self.model_id = f"Qwen/Qwen3.5-{model_size}-Instruct"
        self.data_dir = Path(data_dir)
        self.splits_dir = self.data_dir / "splits"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()

    # --- model init -------------------------------------------------------------------------

    def _load_model(self):
        logging.info(f"> Loading {self.model_id} on {self.device} for evaluation...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        logging.info("\t >>> QWEN3.5 model for [zero-shot/few-shot training] loaded successfully!!!")

    # --- data loading & parsing -------------------------------------------------------------

    def load_and_parse_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Loads and parses the nested BIO-tagged JSONL into flat prompt-ready dictionaries."""
        logging.info(f"> Loading and parsing data from {file_path.name}...")
        parsed_cases = []
        
        if not file_path.exists():
            logging.error(f"\t (!) File not found: {file_path}")
            return parsed_cases
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                raw_record = json.loads(line)
                for case_id, case_data in raw_record.items():
                    parsed_cases.append(self._parse_case(case_id, case_data))
                    
        return parsed_cases

    def _parse_case(self, case_id: str, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translates raw token/label arrays into clean strings and extraction targets."""
        text_lists = case_data.get("text", [])
        label_lists = case_data.get("labels", [])

        sentences = []
        relevance = {}
        premises, claims = [], []

        for i, (tokens, tags) in enumerate(zip(text_lists, label_lists)):
            sentence_str = " ".join(tokens).replace(" ,", ",").replace(" .", ".").replace(" :", ":")
            sentences.append(sentence_str)

            # [SUBTASK 1 - RELEVANCE] (true if any token is a Premise or Claim)
            is_relevant = any(tag != "O" for tag in tags)
            relevance[str(i)] = is_relevant

            # [SUBTASK 1 - SPANS] (premises or claims)
            current_span, current_type = [], None

            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    if current_span:
                        span_str = " ".join(current_span).replace(" ,", ",").replace(" .", ".").replace(" :", ":")
                        if current_type == "Premise": premises.append(span_str)
                        elif current_type == "Claim": claims.append(span_str)
                    
                    current_span = [token]
                    current_type = tag.split("-")[1] 
                
                elif tag.startswith("I-") and current_type == tag.split("-")[1]:
                    current_span.append(token)
                    
                elif tag == "O":
                    if current_span:
                        span_str = " ".join(current_span).replace(" ,", ",").replace(" .", ".").replace(" :", ":")
                        if current_type == "Premise": premises.append(span_str)
                        elif current_type == "Claim": claims.append(span_str)
                        current_span, current_type = [], None

            if current_span:
                span_str = " ".join(current_span).replace(" ,", ",").replace(" .", ".").replace(" :", ":")
                if current_type == "Premise": premises.append(span_str)
                elif current_type == "Claim": claims.append(span_str)

        return {
            "id": case_id,
            "text": sentences,
            "relevance": relevance,
            "premises": premises,
            "claims": claims
        }

    # --- response generation -------------------------------------------------------------


    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, temperature: float = 0.1, do_sample: bool = False) -> str:
        """Generates a response using Qwen's chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # --- subtasks prompting -------------------------------------------------------------------------

    def evaluate_subtask_1(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        """** GRACE SUBTASK 1 : Evidence sentence detection **
        Binary classification of sentences into relevant or not relevant."""
        logging.info(f"> Subtask 1 (relevance detection)...")
        results = []
        
        for case in test_data:
            user_prompt = self._s1_fewshot(case, few_shot_examples)
            response = self._generate(prompts.SYSTEM["SUBTASK_1"], user_prompt)
            # key: ordered case id --> prediction: sentence number + relevance label (true/false)
            results.append({"id": case.get("id"), "prediction": response})
            
        return results

    def evaluate_subtask_2(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        """** GRACE SUBTASK 2 : Evidence span detection **
        Exact Boundary extraction of relevant spans into premises/claims."""
        logging.info(f"> Subtask 2 (span detection)...")
        results = []
        
        for case in test_data:
            user_prompt = self._s2_fewshot(case, few_shot_examples)
            response = self._generate(prompts.SYSTEM["SUBTASK_2"], user_prompt)
            # key: ordered case id --> prediction: extracted spans for premises and claims
            results.append({"id": case.get("id"), "prediction": response})
            
        return results

    def evaluate_subtask_3(self, test_relations: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None, max_new_tokens: int = 10):
        """** GRACE SUBTASK 3 : Relation detection **
        Classification of relations between extracted premises and claims."""
        logging.info(f"> Subtask 3 (relation detection)...")
        results = []
        
        for relation in test_relations:
            user_prompt = self._build_s3_prompt(relation, few_shot_examples)
            response = self._generate(prompts.SYSTEM["SUBTASK_3"], user_prompt, max_new_tokens=max_new_tokens)
            
            results.append({"id": relation.get("id"), "prediction": response.strip()})
            
        return results

    # --- prompt building -------------------------------------------------------------------------
    
    def _s1_fewshot(self, case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
        """Constructs the prompt text for Subtask 1 (sentence relevance classification)."""
        prompt = ""
        
        if examples:
            prompt += "--- EJEMPLOS ---\n"
            for ex in examples:
                prompt += f"Caso clínico:\n{ex.get('text', '')}\n"
                prompt += f"Output esperado (JSON):\n{json.dumps(ex.get('relevance', {}), ensure_ascii=False)}\n\n"
            prompt += "--- FIN DE EJEMPLOS ---\n\n"
            
        prompt += "A continuación el caso clínico a clasificar:\n\n"
        
        sentences = case.get('text', [])
        if isinstance(sentences, list):
            for i, sent in enumerate(sentences):
                sent_str = " ".join(sent) if isinstance(sent, list) else sent
                prompt += f"[{i}] {sent_str}\n"
        else:
            prompt += f"{sentences}\n"
            
        prompt += "\nDevuelve el JSON con las clasificaciones (true/false) para cada oración:"
        return prompt

    def _s2_fewshot(self, case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
        """Constructs the prompt text for Subtask 2 (premise/claim span extraction)."""
        prompt = ""
        
        if examples:
            prompt += "--- EJEMPLOS ---\n"
            for ex in examples:
                prompt += f"Caso clínico:\n{ex.get('text', '')}\n"
                prompt += f"Premisas extraídas:\n- " + "\n- ".join(ex.get('premises', [])) + "\n"
                prompt += f"Claims Extraídas:\n- " + "\n- ".join(ex.get('claims', [])) + "\n\n"
            prompt += "--- FIN DE EJEMPLOS ---\n\n"
            
        prompt += "A continuación se presenta el caso clínico:\n\n"
        text = case.get('text', '')
        if isinstance(text, list):
            text = " ".join([" ".join(s) if isinstance(s, list) else s for s in text])
            
        prompt += f"{text}\n\n"
        prompt += "Extrae los límites exactos de texto. Devuelve el resultado con el siguiente formato:\nPremisas:\n- [span 1]\n- [span 2]\n\nAfirmaciones:\n- [span 3]"
        return prompt

    def _build_s3_prompt(self, relation: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
        """Constructs the prompt text for Subtask 3 (Relation Classification)."""
        prompt = ""
        
        if examples:
            prompt += "--- EJEMPLOS ---\n"
            for ex in examples:
                prompt += f"Premise: \"{ex.get('head', '')}\"\n"
                prompt += f"Claim: \"{ex.get('tail', '')}\"\n"
                prompt += f"Relación: {ex.get('label', '')}\n\n"
            prompt += "--- FIN DE EJEMPLOS ---\n\n"
            
        prompt += "Clasifica la siguiente relación:\n\n"
        prompt += f"Premise: \"{relation.get('head', '')}\"\n"
        prompt += f"Claim: \"{relation.get('tail', '')}\"\n"
        prompt += "Relación:"
        return prompt