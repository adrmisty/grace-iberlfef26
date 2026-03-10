# model.py
# ----------------------------------------------------------------------------------------
# zero-shot & few-shot evaluation on Qwen3.5 for CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import json
import torch
import logging
import config as settings, prompts
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")

class QwenEvaluator:
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
        logging.info(f"> Loading {self.model_id} on {self.device} for training...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        logging.info("\t >>> Model loaded successfully.")

    def _get_response(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
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
                temperature=0.1,
                do_sample=False,
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
            # TODO: zero-shot/few-shot prompts
            user_prompt = self._build_s1_prompt(case, few_shot_examples)
            response = self._get_response(prompts.SYSTEM_PROMPTS["SUBTASK_1"], user_prompt)
            results.append({"id": case.get("id"), "prediction": response})
            
        return results

    def evaluate_subtask_2(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        """** GRACE SUBTASK 2 : Evidence span detection **
        Exact Boundary extraction of relevant spans into premises/claims."""
        logging.info(f"> Subtask 2 (span detection)...")
        raise NotImplementedError("SUBTASK 2.")

    def evaluate_subtask_3(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        """** GRACE SUBTASK 3 : Relation detection **
        Classification of relations between extracted premises and claims."""
        logging.info(f"> Subtask 3 (relation Detection)...")
        raise NotImplementedError("SUBTASK 3.")

    # --- prompt building -------------------------------------------------------------------------
    
    def _build_s1_prompt(self, case: Dict[str, Any], examples: Optional[List[Dict[str, Any]]]) -> str:
        """Constructs the prompt text for Subtask 1."""
        prompt = ""
        if examples:
            prompt += "Ejemplos:\n"
            # TODO: Format few-shot examples
            raise NotImplementedError("SUBTASK 3.")
        
        prompt += f"Caso clínico:\n{case.get('text', '')}\n\nClasifica cada oración:"
        return prompt
    
    # and so on for the rest of the tasks
    # TODO: implement all prompting for each subtask, sampling of examples from train_es...