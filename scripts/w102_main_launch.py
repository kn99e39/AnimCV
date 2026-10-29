"""Process-local W102 launcher; does not change Main's packages or account settings."""
import os
from pathlib import Path
import runpy
import sys

base=Path("C:/Users/dna10/animcv-w102")
for key,relative in {"TMP":"tmp","TEMP":"tmp","TMPDIR":"tmp","XDG_CACHE_HOME":"cache","TORCH_HOME":"cache/torch",
                     "MPLCONFIGDIR":"cache/matplotlib","CUDA_CACHE_PATH":"cache/cuda","HF_HOME":"cache/huggingface"}.items():
    path=base/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.path[:0]=[str(base/"code/src"),str(base/"code/scripts")]
os.chdir(base/"code")
script=base/"code/scripts"/sys.argv[1]
sys.argv=[str(script)]+sys.argv[2:]
runpy.run_path(str(script),run_name="__main__")
