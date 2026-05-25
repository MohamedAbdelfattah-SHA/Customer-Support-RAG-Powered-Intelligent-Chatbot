"""
Read CityFoam RAG experiment results from MLflow.

This module is used by the FastAPI dashboard endpoint.
It does not run experiments and does not modify the production ChromaDB.
"""

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "CityFoam_RAG_Variation_Experiments"
)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_rag_experiment_results() -> dict:
    """
    Loads RAG experiment runs from MLflow and returns dashboard-ready JSON.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        return {
            "experiment_name": EXPERIMENT_NAME,
            "tracking_uri": MLFLOW_TRACKING_URI,
            "runs": [],
            "summary": {
                "total_runs": 0,
                "best_config": None,
                "best_hit_rate": 0,
                "best_rag_score": 0,
                "lowest_poor_rate": 0,
            },
        }

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.hit_rate DESC"],
        max_results=500,
    )

    rows = []

    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags

        # Ignore parent runs that do not represent a single configuration.
        if "hit_rate" not in metrics:
            continue

        row = {
            "run_id": run.info.run_id,
            "run_name": tags.get("mlflow.runName", ""),
            "chunk_size": params.get("chunk_size", "-"),
            "chunk_overlap": params.get("chunk_overlap", "-"),
            "n_results": params.get("n_results", "-"),
            "hit_rate": _to_float(metrics.get("hit_rate")),
            "mean_rag_score": _to_float(metrics.get("mean_rag_score")),
            "mean_decay_score": _to_float(metrics.get("mean_decay_score")),
            "pct_poor": _to_float(metrics.get("pct_poor")),
            "mean_top1_dist": _to_float(metrics.get("mean_top1_dist")),
            "n_chunks": _to_float(metrics.get("n_chunks")),
            "ingest_secs": _to_float(metrics.get("ingest_secs")),
        }

        rows.append(row)

    rows.sort(
        key=lambda x: (
            x["hit_rate"],
            x["mean_rag_score"],
            -x["pct_poor"],
            -x["mean_top1_dist"],
        ),
        reverse=True,
    )

    if not rows:
        best = None
        lowest_poor = 0
    else:
        best = rows[0]
        lowest_poor = min(row["pct_poor"] for row in rows)

    return {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "runs": rows,
        "summary": {
            "total_runs": len(rows),
            "best_hit_rate": best["hit_rate"] if best else 0,
            "best_rag_score": best["mean_rag_score"] if best else 0,
            "lowest_poor_rate": lowest_poor,
            "best_config": {
                "run_name": best["run_name"],
                "chunk_size": best["chunk_size"],
                "chunk_overlap": best["chunk_overlap"],
                "n_results": best["n_results"],
            } if best else None,
        },
    }