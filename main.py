import os
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

# ── Print CUDA + feature-layout diagnostics BEFORE any heavy imports ──
# This makes it immediately obvious whether the run will use GPU or CPU,
# and shows exactly which features are active for this experiment.
from config import print_startup_diagnostics, config
print_startup_diagnostics()

from training import train_model
from inference import inference_main
from unified_framework import unified_framework
from route_evaluation import plot_model_overview_performance
from extract_features import main as extract_embeddings_main
from lda_classifier import train_lda, evaluate_lda
from mlp_classifier import train_mlp, evaluate_mlp
from xgboost_classifier import train_xgboost, evaluate_xgboost
from compare_classifiers import main as compare_classifiers_main


def main():
    print("\n=== Starting PSTAR Model for Path Planning Transformer ===")


    if config.get("pstar_training", False):
        print("\n=== Starting PSTAR Training ===")
        train_model()

    if config.get("inference", False):
        print("\n=== Starting Inference ===")
        inference_main()

    if config.get("unified_framework", False):
        print("\n=== Starting Unified Framework ===")
        unified_framework()

    if config.get("ragate_training", False):
        print("\n=== Starting RAGate Training ===")

        print("\n-- Step 1: Extracting context embeddings --")
        extract_embeddings_main()

        print("\n-- Step 2: Training LDA classifier --")
        train_lda()

        print("\n-- Step 3: Evaluating LDA classifier --")
        evaluate_lda()

        print("\n-- Step 4: Training MLP classifier --")
        train_mlp()

        print("\n-- Step 5: Evaluating MLP classifier --")
        evaluate_mlp()

        print("\n-- Step 6: Training XGBoost classifier --")
        train_xgboost()

        print("\n-- Step 7: Evaluating XGBoost classifier --")
        evaluate_xgboost()

        print("\n-- Step 8: Comparing classifiers --")
        compare_classifiers_main()

        print("\n=== RAGate Model Training Complete ===")

    print("\n==== Execution Complete ====")


if __name__ == "__main__":
    main()
