import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import subprocess
import importlib.util

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "1200000"

def _parse_requirements(requirements_path):
    requirements = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "--")):
                continue
            name = line.split(";", 1)[0].strip()
            for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
                if sep in name:
                    name = name.split(sep, 1)[0].strip()
                    break
            if "[" in name:
                name = name.split("[", 1)[0].strip()
            if name:
                requirements.append(name)
    return requirements

def ensure_requirements_installed():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.isfile(requirements_path):
        return
    package_names = _parse_requirements(requirements_path)
    if not package_names:
        return
    module_aliases = {"pillow": "PIL", "scikit-learn": "sklearn"}
    missing = [
        pkg for pkg in package_names
        if importlib.util.find_spec(module_aliases.get(pkg, pkg)) is None
    ]
    if not missing:
        return
    print(f"Installing missing packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

ensure_requirements_installed()

from config import print_startup_diagnostics, config
print_startup_diagnostics()

from training import train_model
from inference import inference_main
from unified_framework import unified_framework
from route_evaluation import plot_model_overview_performance
from extract_features import main as extract_embeddings_main
from lda_classifier import train_lda, evaluate_lda
from mlp_classifier import train_mlp, evaluate_mlp, evaluate_finetune_head
from xgboost_classifier import train_xgboost, evaluate_xgboost
from compare_classifiers import main as compare_classifiers_main

def main():
    from run_report import StageTimer, write_report
    from utils import set_seed
    print("\n=== Starting PSTAR Model for Path Planning Transformer ===")
    print(f"--> Architecture: {config.get('architecture', 'pstar')}")
    set_seed()  

    if config.get("pstar_training", False):
        print("\n=== Starting PSTAR Training ===")
        with StageTimer(config, "PSTAR_training"):
            train_model()

    if config.get("inference", False):
        print("\n=== Starting Inference ===")
        with StageTimer(config, "inference"):
            inference_main()

    if config.get("ragate_training", False):
        print("\n=== Starting RAGate Training ===")

        if config.get("ragate_joint_finetune", False):
            print("\n-- Step 0: Joint fine-tuning PSTAR + MLP --")
            _ft_ckpt_path  = config["ragate_joint_finetune_checkpoint"]
            _force_ft      = config.get("ragate_embeddings_force_reextract", False)
            _ft_exists     = os.path.exists(_ft_ckpt_path)
            if _ft_exists and not _force_ft:
                print(f"   [SKIP] Fine-tuned checkpoint already exists: {_ft_ckpt_path}")
                print(f"          Set ragate_embeddings_force_reextract=True to re-run Step 0.")
            else:
                from mlp_classifier import finetune_transformer
                with StageTimer(config, "RAGate_joint_finetune"):
                    finetune_transformer()

            _ft_ckpt = config["ragate_joint_finetune_checkpoint"]
            config["inference_model_name"] = os.path.basename(_ft_ckpt)
            config["checkpoint_dir"]       = os.path.dirname(os.path.abspath(_ft_ckpt))

            if config.get("ragate_step0_eval_head", True):
                print("\n-- Step 0b: Evaluating fine-tuned PSTAR + MLP head --")
                with StageTimer(config, "RAGate_step0b_eval"):
                    evaluate_finetune_head()
            else:
                print("-- Step 0b skipped (ragate_step0_eval_head=False) --")

            print(f"\n-- Step 0 complete.")
            print(f"   PSTAR now points to: {config['inference_model_name']}")
            print(f"   Checkpoint dir:      {config['checkpoint_dir']}")
            if not _force_ft:
                print("   [WARNING] ragate_embeddings_force_reextract=False in config.py —"
                      " Step 1 will skip re-extraction and use stale embeddings.")

        steps = [
            ("RAGate_embedding_extraction", "Step 1: Extracting context embeddings", extract_embeddings_main),
            ("RAGate_LDA_train",     "Step 2: Training LDA classifier",       train_lda),
            ("RAGate_LDA_eval",      "Step 3: Evaluating LDA classifier",     evaluate_lda),
            ("RAGate_MLP_train",     "Step 4: Training MLP classifier",       train_mlp),
            ("RAGate_MLP_eval",      "Step 5: Evaluating MLP classifier",     evaluate_mlp),
            ("RAGate_XGBoost_train", "Step 6: Training XGBoost classifier",   train_xgboost),
            ("RAGate_XGBoost_eval",  "Step 7: Evaluating XGBoost classifier", evaluate_xgboost),
            ("RAGate_compare",       "Step 8: Comparing classifiers",         compare_classifiers_main),
        ]
        for stage, label, fn in steps:
            print(f"\n-- {label} --")
            with StageTimer(config, stage):
                fn()
        print("\n=== RAGate Model Training Complete ===")

    if config.get("unified_framework", False):
        print("\n=== Starting Unified Framework ===")
        with StageTimer(config, "unified_framework"):
            unified_framework()

    if config.get("timing_benchmark", False):
        print("\n=== Starting Timing Benchmark ===")
        from timing_harness import timing_main
        with StageTimer(config, "timing_benchmark"):
            timing_main()

    if config.get("write_run_report", True):
        try:
            write_report(config)
        except Exception as e:                    
            print(f"\n  [REPORT] could not be written: {type(e).__name__}: {e}")

    print("\n==== Execution Complete ====")

if __name__ == "__main__":
    main()
