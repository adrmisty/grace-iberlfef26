# main.py
# ----------------------------------------------------------------------------------------
# relation mapping and alignment for CasiMedicos-Arg (GRACE / IBERLEF26)
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# mar-2026

import logging
import config as settings
from relations import align
from eval import evaluate
    

if __name__ == "__main__":
    #align()
    evaluate()