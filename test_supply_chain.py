"""
End-to-end verification run on invoices.csv → predict payment_status.
Prints every stage of output so we can confirm quality.
"""
import json
import textwrap

from main import determine_task_type
from preprocessing import preprocess_data
from model_selection import select_model
from explainability import get_top_features, build_explanation
import joblib

CSV = "invoices.csv"
TARGET = "payment_status"

# ── 1. Task detection ───────────────────────────────────────────────────────
task_type = determine_task_type(CSV, TARGET)
print(f"\n{'='*60}")
print(f"TASK TYPE: {task_type}")
print(f"{'='*60}")

# ── 2. Preprocessing + missing value report ─────────────────────────────────
X, y, mv_report = preprocess_data(CSV, TARGET)

print(f"\nPREPROCESSED SHAPE: {X.shape}")
print(f"\nMISSING VALUE REPORT ({len(mv_report)} columns):")
print(f"{'Column':<20} {'Strategy':<22} {'Missing':<10} Reason")
print("-" * 90)
for col, info in mv_report.items():
    flagged = " ⚑" if info.get("flagged") else ""
    print(f"{col:<20} {info['strategy']+flagged:<22} {info['missing_rate']:<10} {info['reason']}")

# ── 3. Model selection + tuning + overfitting ───────────────────────────────
print(f"\n{'='*60}")
print("MODEL SELECTION + TUNING")
print(f"{'='*60}")
result = select_model(task_type, X, y)

print(f"\nAll model CV scores:")
for name, score in sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True):
    marker = " << WINNER" if name == result["model_name"] else ""
    print(f"  {name:<30} {score:.4f}{marker}")

print(f"\nBest model (after tuning): {result['model_name']}")
print(f"CV score (final):          {result['cv_score_final']:.4f}")
print(f"Train score:               {result['train_score']:.4f}")
print(f"Overfitting risk:          {result['overfitting_risk'].upper()}")
print(f"  (train - cv = {result['train_score'] - result['cv_score_final']:.4f})")

joblib.dump(result["model"], "best_model.pkl")

# ── 4. SHAP + explanation ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("EXPLAINABILITY (SHAP)")
print(f"{'='*60}")
top_features = get_top_features(result["model"], X, n=3)
print(f"\nTop 3 features by SHAP importance: {top_features}")

explanation = build_explanation(
    model_name=result["model_name"],
    all_scores=result["all_scores"],
    top_features=top_features,
    cv_score=result["cv_score_final"],
    task_type=task_type,
)
print(f"\nNATURAL LANGUAGE EXPLANATION:")
print(textwrap.fill(explanation, width=80))

# ── 5. Full orchestrator output (what the MCP tool returns) ─────────────────
print(f"\n{'='*60}")
print("FULL auto_ml_pipeline_tool OUTPUT (JSON)")
print(f"{'='*60}")
output = {
    "task_type": task_type,
    "best_model": result["model_name"],
    "cv_score": round(result["cv_score_final"], 4),
    "train_score": round(result["train_score"], 4),
    "overfitting_risk": result["overfitting_risk"],
    "top_features": top_features,
    "explanation": explanation,
    "missing_value_report": mv_report,
}
print(json.dumps(output, indent=2))
