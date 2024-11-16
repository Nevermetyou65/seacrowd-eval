import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from huggingface_hub import snapshot_download

def read_all_pending_model(EVAL_REQUESTS_PATH: str) -> Dict[str, List[Tuple[Any, str]]]:
    depth = 1
    alls = defaultdict(list)
    for root, _, files in os.walk(EVAL_REQUESTS_PATH):
        current_depth = root.count(os.sep) - EVAL_REQUESTS_PATH.count(os.sep)
        if current_depth == depth:
            for file in files:
                if not file.endswith(".json"):
                    continue
                file_abs_path = os.path.join(root, file)
                with open(file_abs_path, "r") as f:
                    info = json.load(f)
                    alls[info['model']].append((info, file_abs_path))

    pendings = {}
    for k in alls.keys():
        is_pending = False
        for stat in alls[k]:
            info_dict = stat[0]
            if info_dict['status'] == "PENDING":
                is_pending = True
        if is_pending:
            pendings[k] = alls[k]
    return pendings
                
def read_model_for_name(model_name: str, EVAL_REQUESTS_PATH: str) -> List[Tuple[Any, str]]:
    depth = 1
    alls = defaultdict(list)
    for root, _, files in os.walk(EVAL_REQUESTS_PATH):
        current_depth = root.count(os.sep) - EVAL_REQUESTS_PATH.count(os.sep)
        if current_depth == depth:
            for file in files:
                if not file.endswith(".json"):
                    continue
                file_abs_path = os.path.join(root, file)
                with open(file_abs_path, "r") as f:
                    info = json.load(f)
                    alls[info['model']].append((info, file_abs_path))
    return alls[model_name]

def read_specific_result(task: str, model_name: str):
    CACHE_PATH = os.getenv("HF_HOME", ".")
    RESULTS_REPO = os.environ["RESULTS_REPO"]
    EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
    snapshot_download(
        repo_id=RESULTS_REPO,
        local_dir=EVAL_RESULTS_PATH,
        repo_type="dataset",
        tqdm_class=None,
        etag_timeout=30,
    )
    with open(f'{EVAL_RESULTS_PATH}/{task}/{model_name}/result.json') as f:
        data = json.load(f)
    return data