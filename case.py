# case.py
# ----------------------------------------------------------------------------------------
# clinical case parser for https://huggingface.co/datasets/HiTZ/casimedicos-arg
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

from collections import defaultdict
from enum import Enum, auto
from typing import Dict, Set, List

class CaseParser:
    """Parser for the clinical cases in the CasiMedicos dataset."""
    
    class CaseBlock(Enum):
        HEADER = auto(); NARRATIVE = auto(); EXPLANATION = auto()
    
    def __init__(self, delimiter: str = " "):
        self.delimiter = delimiter
        self.case_header = "CLINICAL CASE:"
        self.end_header = "CORRECT ANSWER:"

    def parse(self, filepath: str) -> Dict[str, List[str]]:
        """Extracts sentences from the clinical case descriptions, removing annotations."""
        cases = defaultdict(list)
        current_case_id = ""
        current_sentence = []
        case_counter = 0 
        
        current_state = self.CaseBlock.HEADER
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if self.case_header in line:
                    case_counter += 1
                    current_case_id = f"case_{case_counter}"
                    current_state = self.CaseBlock.NARRATIVE
                    continue
                    
                if self.end_header in line:
                    current_state = self.CaseBlock.EXPLANATION
                    continue
                
                if not line:
                    if current_sentence and current_state == self.CaseBlock.NARRATIVE:
                        reconstructed = " ".join(current_sentence)
                        reconstructed = re.sub(r'\s+([.,;:?!)])', r'\1', reconstructed)
                        cases[current_case_id].append(reconstructed)
                        current_sentence = []
                    
                    if current_state == self.CaseBlock.EXPLANATION:
                        current_state = self.CaseBlock.HEADER
                        
                    continue
                
                parts = line.split(self.delimiter)
                if len(parts) >= 1 and current_state == self.CaseBlock.NARRATIVE:
                    token = parts[0]
                    if not re.match(r'^\d+-$', token):
                        current_sentence.append(token)
                        
        if current_sentence and current_state == self.CaseBlock.NARRATIVE:
            cases[current_case_id].append(" ".join(current_sentence))
            
        return cases
    
    def get_qa_themes(self, filepath: str, delimiter: str = " ") -> Set[str]:
        """Extracts unique question types ('HEMATOLOGY', 'PEDIATRICS', 'PALLIATIVE CARE'..)."""
        
        themes = set()
        is_reading_theme = False
        current_theme = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(delimiter)
                if not parts:
                    continue
                    
                token = parts[0]
                
                if token == "TYPE:":
                    is_reading_theme = True
                    current_theme = []
                    continue # 'CLINICAL CASE' right after type
                if token == "CLINICAL" and is_reading_theme:
                    is_reading_theme = False
                    if current_theme:
                        themes.add(" ".join(current_theme))
                    continue
                if is_reading_theme:
                    current_theme.append(token)
                    
        return themes