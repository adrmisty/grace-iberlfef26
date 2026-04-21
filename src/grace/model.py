# model.py
# ----------------------------------------------------------------------------------------
# zero-shot & few-shot run on LLMs for CasiMedicos-Arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import os
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
from google.generativeai import types as genaitypes
from openai import OpenAI
import re
import src.config as settings
import src.grace.prompts as prompts

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
        """Generates model's response to the zero/few-shot prompt for local HuggingFace models."""
        messages = self._build_messages(system_prompt, user_prompt)
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
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
        
        decoded_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if isinstance(decoded_text, list):
            decoded_text = decoded_text[0]
        return prefill + decoded_text

    def _load_model(self):
        raise NotImplementedError("> impl. in model subclass")
    
    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        raise NotImplementedError("> impl. in model subclass")

    # --- subtasks prompting -------------------------------------------------------------------------

# --- subtasks prompting -------------------------------------------------------------------------

    def run_subtask_1(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None, lang: str = settings.LANG):
        logging.info(f"> Subtask 1 (relevance detection)...")
        results = []
        for case in test_data:
            user_prompt = prompts.build_s1_prompt(case, few_shot_examples, lang=lang)
            sys_prompt = prompts.SYSTEM_PROMPTS[lang]["SUBTASK_1"]
            response = self._generate(sys_prompt, user_prompt, max_new_tokens=2048, prefill="{\n")
            results.append({"id": case.get("id"), "prediction": response})
        return results

    def run_subtask_2(self, test_data: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None, lang: str = settings.LANG):
        logging.info(f"> Subtask 2 (span detection)...")
        results = []
        for case in test_data:
            user_prompt = prompts.build_s2_prompt(case, few_shot_examples, lang=lang)
            sys_prompt = prompts.SYSTEM_PROMPTS[lang]["SUBTASK_2"]
            #  prefill from "Premisas:\n-" to "{\n" to enforce the strict JSON output
            response = self._generate(sys_prompt, user_prompt, max_new_tokens=2048, prefill="{\n")
            results.append({"id": case.get("id"), "prediction": response})
        return results

    def run_subtask_3(self, test_relations: List[Dict[str, Any]], few_shot_examples: Optional[List[Dict[str, Any]]] = None, max_new_tokens: int = 128, lang: str = settings.LANG):
        logging.info(f"> Subtask 3 (relation detection)...")
        results = []
        for relation in test_relations:
            user_prompt = prompts.build_s3_prompt(relation, few_shot_examples, lang=lang)
            sys_prompt = prompts.SYSTEM_PROMPTS[lang]["SUBTASK_3"]
            response = self._generate(sys_prompt, user_prompt, max_new_tokens=max_new_tokens, prefill='{\n  "label": "')
            results.append({"id": relation.get("id"), "prediction": response.strip()})
        return results

# --- hugging face models -------------------------------------------------------------------------

class GraceModel(Model):
    def __init__(self, model_size: str = "2B", data_dir: str = settings.BASE_DATA_DIR):
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
# for this, do not use the prefill trick

class GeminiAPIModel(Model):
    def __init__(self, model_version: str = "gemini-1.5-flash", data_dir: str = settings.BASE_DATA_DIR):
        super().__init__(model_version, data_dir)
        self.model_id = model_version
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
        pass

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 2048, prefill: str = "") -> str:
        if "{" in prefill:
            instruction = "REGLA ABSOLUTA: Tu respuesta debe ser ÚNICAMENTE un JSON válido. NINGUNA palabra antes o después. NO uses formato Markdown, solo el JSON puro."
        else:
            instruction = "REGLA ABSOLUTA: Tu respuesta debe comenzar directamente con la palabra 'Premisas:'. NINGUNA palabra antes o después. NO uses formato JSON, usa estrictamente listas con guiones (-)."
            
        prompt = f"INSTRUCCIONES DEL SISTEMA:\n{system_prompt}\n\nENTRADA DEL USUARIO:\n{user_prompt}\n\n{instruction}"
        
        safety_settings = {
            genaitypes.HarmCategory.HARM_CATEGORY_HARASSMENT: genaitypes.HarmBlockThreshold.BLOCK_NONE,
            genaitypes.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genaitypes.HarmBlockThreshold.BLOCK_NONE,
            genaitypes.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genaitypes.HarmBlockThreshold.BLOCK_NONE,
            genaitypes.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genaitypes.HarmBlockThreshold.BLOCK_NONE,
        }
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0.0,
                )
            )

            try:
                text = response.text.strip()
            except ValueError:
                return ""

            text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^```\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"\t> (!) Gemini API error during generation: {e}")
            return ""
                    
class OpenAIModel(Model):
    def __init__(self, model_version: str = "gpt-5.4-mini", data_dir: str = settings.BASE_DATA_DIR):
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
        if "{" in prefill:
            instruction = "Tu respuesta debe ser ÚNICAMENTE un JSON válido. NINGUNA palabra antes o después. NO uses comillas invertidas de markdown (```)."
        else:
            instruction = "Tu respuesta debe comenzar directamente con la palabra 'Premisas:'. NINGUNA palabra antes o después. NO uses formato JSON, usa estrictamente listas con guiones (-)."

        system_prompt += f"\n\n{instruction}"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=max_new_tokens,
                temperature=0.0
            )
            
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^```\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"\t> (!) OpenAI API error during generation: {e}")
            return ""

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