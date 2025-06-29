# tests/test_evaluation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from iris import evaluate_model

def test_evaluate_model_returns_metrics():
    metrics = evaluate_model()
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.5
