# model.py
# ----------------------------------------------------------------------------------------
# zero-shot & few-shot run on LLMs for CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import json
import os
import torch
import logging
import src.config as settings
import src.grace.prompts as prompts
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")


class Model:
    def __init__(self, model_size: str, data_dir: str):
        """Inits the model for a given size."""
        self.data_dir = Path(data_dir)
        self.splits_dir = self.data_dir / "splits"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = ""
        self.model = None
        self.model_size = model_size
        self.tokenizer = None

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, prefill: str = "") -> str:
        """Generates model's response to the zero/few-shot prompt."""
        messages = self._build_messages(system_prompt, user_prompt)
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if prefill: # instruct to reinforce JSON/list results
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
        
        decoded_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if isinstance(decoded_text, list):
            decoded_text = decoded_text[0]
        return prefill + decoded_text

    def _load_model(self):
        raise NotImplementedError("> impl. in model subclass")
    
    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        raise NotImplementedError("> impl. in model subclass")

    # --- subtasks prompting -------------------------------------------------------------------------

    def run_subtask_1(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        logging.info(f"> Subtask 1 (relevance detection)...")
        results = []
        for case in test_data:
            user_prompt = prompts.build_s1_prompt(case, few_shot_examples)
            response = self._generate(prompts.SYSTEM_PROMPT["SUBTASK_1"], user_prompt, max_new_tokens=1024, prefill="{\n")
            results.append({"id": case.get("id"), "prediction": response})
        return results

    def run_subtask_2(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None):
        logging.info(f"> Subtask 2 (span detection)...")
        results = []
        for case in test_data:
            user_prompt = prompts.build_s2_prompt(case, few_shot_examples)
            response = self._generate(prompts.SYSTEM_PROMPT["SUBTASK_2"], user_prompt, max_new_tokens=1024, prefill="Premisas:\n-")
            results.append({"id": case.get("id"), "prediction": response})
        return results

    def run_subtask_3(self, test_relations: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None, max_new_tokens: int = 128): # INCREASED DEFAULT TO 128
        logging.info(f"> Subtask 3 (relation detection)...")
        results = []
        for relation in test_relations:
            user_prompt = prompts.build_s3_prompt(relation, few_shot_examples)
            response = self._generate(prompts.SYSTEM_PROMPT["SUBTASK_3"], user_prompt, max_new_tokens=max_new_tokens, prefill='{\n  "label": "')
            results.append({"id": relation.get("id"), "prediction": response.strip()})
        return results

# --- hugging face models -------------------------------------------------------------------------

class GraceModel(Model):
    def __init__(self, model_size: str = "2B", data_dir: str = settings.BASE_DATA_DIR):
        """Inits the Qwen model."""
        super().__init__(model_size, data_dir)
        self.model_id = f"Qwen/Qwen3.5-{model_size}"
        self._load_model()

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
        logging.info(f"\t >>> {self.model_id} model loaded successfully!!!")

    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class MedGemmaModel(Model):
    def __init__(self, model_size: str = "4B", data_dir: str = settings.BASE_DATA_DIR):
        """Inits the MedGemma model."""
        super().__init__(model_size, data_dir)
        self.model_id = f"google/medgemma-{model_size.lower()}-it"
        self._load_model()

    def _load_model(self):
        logging.info(f"> Loading {self.model_id} on {self.device} for evaluation...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        logging.info(f"\t >>> {self.model_id} model loaded successfully!!!")

    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
        ]

# --- gemini/openai APIs -------------------------------------------------------------------------

class GeminiAPIModel(Model):
    def __init__(self, model_version: str = "gemini-3-flash-preview", data_dir: str = settings.BASE_DATA_DIR):
        """Inits the Gemini API client."""
        super().__init__(model_version, data_dir)
        self.model_id = model_version
        print(self.model_id)
        pass
        self._load_model()

    def _load_model(self):
        logging.info(f"> Initializing API client for {self.model_id}...")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("> (!) GEMINI_API_KEY missing")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_id)
        logging.info(f"\t >>> {self.model_id} API client loaded successfully!!!")

    def _build_messages(self, system_prompt: str, user_prompt: str):
        """Unimplemented for Gemini (prompt handling in _generate)."""
        pass

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, prefill: str = "") -> str:
        """Overrides the local HF generation to use Google's API."""
        prompt = f"INSTRUCCIONES DEL SISTEMA:\n{system_prompt}\n\nENTRADA DEL USUARIO:\n{user_prompt}\n\n{prefill}"
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0.0,  # greedy decoding equivalent
                )
            )

            try:
                text = response.text.strip()
                text = text.replace("```json", "").replace("```", "").strip()
            except ValueError:
                reason = response.candidates.finish_reason.name if response.candidates else "Unknown"
                logging.error(f"\t> (!) Gemini API returned empty text: {reason}")
                return prefill

            if text.startswith(prefill.strip()):
                text = text[len(prefill.strip()):].strip()
                
            return prefill + text
            
        except Exception as e:
            logging.error(f"\t> (!) Gemini API error during generation: {e}")
            return prefill

class OpenAIModel(Model):
    def __init__(self, model_version: str = "gpt-5.4-mini", data_dir: str = settings.BASE_DATA_DIR):
        """Inits the OpenAI API client. 
        ** use GPT similar in size and pricing to gemini-3-flash! (5.4-mini, 5.4-nano...)."""
        super().__init__(model_version, data_dir)
        self.model_id = model_version
        self._load_model()

    def _load_model(self):
        logging.info(f"> Initializing API client for {self.model_id}...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(">(!) OPENAI_API_KEY missing")
            
        self.client = OpenAI(api_key=api_key)
        logging.info(f"\t >>> {self.model_id} API client loaded successfully!!!")

    def _build_messages(self, system_prompt: str, user_prompt: str):
        pass 

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, prefill: str = "") -> str:
        """Overrides the local HF generation to use OpenAI's API."""
        if prefill:
            user_prompt += f"\n\nEmpieza tu respuesta EXACTAMENTE con la siguiente cadena y nada más:\n{prefill}"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0
            )
            
            text = response.choices.message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            
            if text.startswith(prefill.strip()):
                text = text[len(prefill.strip()):].strip()
                
            return prefill + text
            
        except Exception as e:
            logging.error(f"\t> (!) OpenAI API error during generation: {e}")
            return prefill

# --- model factory -------------------------------------------------------------------------

MODEL_FACTORY = {
    "qwen": {"class": GraceModel, "prefix": "Qwen"},
    "medgemma": {"class": MedGemmaModel, "prefix": "MedGemma"},
    "gemini": {"class": GeminiAPIModel, "prefix": "Gemini"},
    "openai": {"class": OpenAIModel, "prefix": "OpenAI"},
}

def get_model(model_type: str, model_size: str, data_dir: str):
    """Factory to instantiate the correct model dynamically."""
    config = MODEL_FACTORY.get(model_type.lower())
    if not config:
        raise ValueError(f"Model '{model_type}' not supported.")
    return config["class"](model_size, data_dir), config["prefix"]