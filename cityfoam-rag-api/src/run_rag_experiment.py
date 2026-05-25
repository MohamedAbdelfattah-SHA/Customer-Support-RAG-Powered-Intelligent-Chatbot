"""
CityFoam RAG Hyperparameter Experiment Runner

This script runs offline RAG experiments and logs results to MLflow.

It is intentionally isolated from the production API:
- It does not modify src/api.py
- It does not modify src/ingest.py
- It builds temporary ChromaDB collections only
- It uses the same data folder and preprocessing logic style as the main project

Run from project root:
    python src/run_rag_experiment.py

Open MLflow UI:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
"""

import itertools
import logging
import os
import re
import tempfile
import time
import gc
from pathlib import Path

import chromadb
import mlflow
import pandas as pd
import pymupdf4llm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition

from rag_eval import score_distances


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cityfoam.rag_experiment")


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
EVALUATION_DATASET = Path(
    os.getenv(
        "EVALUATION_DATASET",
        PROJECT_ROOT / "deliverables" / "evaluation_dataset.csv"
    )
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "CityFoam_RAG_Variation_Experiments"
)


# ---------------------------------------------------------
# Experiment Grid
# ---------------------------------------------------------
PARAM_GRID = {
    "chunk_size": [600, 900, 1200],
    "chunk_overlap": [100, 200],
    "n_results": [3, 5, 7],
}


EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


INTERNAL_SYSTEM_FILES = {
    "golden_dataset.xlsx",
    "live_metrics_log.csv",
    "pending_files.json",
    "generated_questions.xlsx",
    "mlflow.db",
    "pso_best_params.json",
    "pso_best_params.tmp.json",
    "user_queries.jsonl",
}


def is_internal_system_file(filename: str) -> bool:
    name = os.path.basename(str(filename)).lower()
    return name in INTERNAL_SYSTEM_FILES

# ---------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------
def normalize_arabic(text: str) -> str:
    """
    Same Arabic normalization logic used in the main ingestion pipeline.
    This is duplicated here to keep the experiment isolated.
    """
    if not text:
        return ""

    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\u064B-\u065F]", "", text)
    return text


# ---------------------------------------------------------
# Isolated Data Processing
# ---------------------------------------------------------
def process_data_folder_for_experiment(
    chunk_size: int,
    chunk_overlap: int
) -> list[dict]:
    """
    Processes the data folder using a chosen chunk_size and chunk_overlap.

    This function is intentionally separate from src/ingest.py so that
    production ingestion remains unchanged.
    """
    logger.info(
        "Processing data with chunk_size=%s, chunk_overlap=%s",
        chunk_size,
        chunk_overlap
    )

    documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    for file_path in DATA_DIR.iterdir():
        if file_path.is_dir() or file_path.name.startswith("."):
            continue
        if is_internal_system_file(file_path.name):
            logger.info("Skipping internal system file: %s", file_path.name)
            continue

        filename = file_path.name
        ext = file_path.suffix.lower().replace(".", "")

        logger.info("Processing file: %s", filename)

        if ext in ["xlsx", "xls", "csv"]:
            try:
                df = pd.read_csv(file_path) if ext == "csv" else pd.read_excel(file_path)

                for _, row in df.fillna("N/A").iterrows():
                    row_text = f"--- Database Record ({filename}) ---\n"
                    row_text += "\n".join([
                        f"{normalize_arabic(str(k))}: {normalize_arabic(str(v))}"
                        for k, v in row.items()
                    ])

                    documents.append({
                        "text": row_text.strip(),
                        "metadata": {
                            "source": filename,
                            "type": "spreadsheet"
                        }
                    })

            except Exception as exc:
                logger.error("Error processing spreadsheet %s: %s", filename, exc)

        elif ext == "pdf":
            try:
                md_text = pymupdf4llm.to_markdown(str(file_path))
                clean_text = normalize_arabic(md_text)
                chunks = text_splitter.split_text(clean_text)

                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "type": "pdf_as_markdown",
                            "chunk_index": i
                        }
                    })

            except Exception as exc:
                logger.error("Error processing PDF %s: %s", filename, exc)

        elif ext in ["docx", "txt"]:
            try:
                elements = partition(filename=str(file_path))
                full_text = "\n\n".join([
                    normalize_arabic(el.text)
                    for el in elements
                    if hasattr(el, "text")
                ])

                chunks = text_splitter.split_text(full_text)

                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "type": "document",
                            "chunk_index": i
                        }
                    })

            except Exception as exc:
                logger.error("Error processing document %s: %s", filename, exc)

    return documents


# ---------------------------------------------------------
# Evaluation Dataset
# ---------------------------------------------------------
def load_evaluation_queries() -> pd.DataFrame:
    """
    Loads the project evaluation dataset.

    Expected columns:
    - query
    - expected_source
    - query_type
    """
    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_DATASET}"
        )

    df = pd.read_csv(EVALUATION_DATASET)

    required_columns = {"query", "expected_source", "query_type"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Evaluation dataset missing columns: {missing_columns}"
        )

    return df


# ---------------------------------------------------------
# Temporary ChromaDB Build
# ---------------------------------------------------------
def build_temp_collection(
    documents: list[dict],
    tmp_dir: str,
    model: SentenceTransformer
):
    """
    Builds a temporary ChromaDB collection for one experiment configuration.
    """
    client = chromadb.PersistentClient(path=tmp_dir)
    collection = client.create_collection(name="cityfoam_experiment_collection")

    texts = [doc["text"] for doc in documents]

    if not texts:
        raise ValueError("No documents were generated from the data folder.")

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    ).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[doc["metadata"] for doc in documents],
        ids=[f"exp_doc_{i}" for i in range(len(documents))]
    )

    return collection


# ---------------------------------------------------------
# Evaluate One Configuration
# ---------------------------------------------------------
def evaluate_configuration(
    collection,
    model: SentenceTransformer,
    eval_df: pd.DataFrame,
    n_results: int
) -> dict:
    """
    Evaluates one retrieval configuration against evaluation_dataset.csv.
    """
    hit_count = 0
    poor_count = 0
    gap_scores = []
    decay_scores = []
    top1_distances = []

    type_hits = {}

    for _, row in eval_df.iterrows():
        query = normalize_arabic(str(row["query"]))
        expected_source = str(row["expected_source"]).strip()
        query_type = str(row["query_type"]).strip()

        query_embedding = model.encode([query]).tolist()

        result = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["distances", "metadatas"]
        )

        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        retrieved_sources = [
            meta.get("source", "")
            for meta in metadatas
            if isinstance(meta, dict)
        ]

        is_hit = expected_source in retrieved_sources

        if is_hit:
            hit_count += 1

        if query_type not in type_hits:
            type_hits[query_type] = {"hits": 0, "total": 0}

        type_hits[query_type]["total"] += 1
        type_hits[query_type]["hits"] += int(is_hit)

        retrieval_score = score_distances(distances)

        gap_scores.append(retrieval_score.gap_score)
        decay_scores.append(retrieval_score.decay_score)

        if retrieval_score.is_poor:
            poor_count += 1

        if distances:
            top1_distances.append(sorted(distances)[0])

    total = len(eval_df)

    metrics = {
        "hit_rate": round((hit_count / total) * 100, 2),
        "mean_rag_score": round(sum(gap_scores) / total, 4),
        "mean_decay_score": round(sum(decay_scores) / total, 4),
        "pct_poor": round((poor_count / total) * 100, 2),
        "mean_top1_dist": round(
            sum(top1_distances) / max(len(top1_distances), 1),
            4
        ),
    }

    for query_type, counts in type_hits.items():
        safe_type = query_type.lower().replace(" ", "_").replace("-", "_")
        metrics[f"hit_rate_{safe_type}"] = round(
            (counts["hits"] / counts["total"]) * 100,
            2
        )

    return metrics


# ---------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------
# ---------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------
def run_experiment():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    eval_df = load_evaluation_queries()

    combinations = list(itertools.product(*PARAM_GRID.values()))
    keys = list(PARAM_GRID.keys())

    logger.info(
        "Running %d configurations against %d evaluation queries.",
        len(combinations),
        len(eval_df)
    )

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    best_config = None
    best_score = -1

    with mlflow.start_run(run_name="cityfoam-rag-variation-sweep") as parent_run:
        mlflow.set_tag("experiment_type", "rag_variation_sweep")
        mlflow.set_tag("embedding_model", EMBEDDING_MODEL_NAME)
        mlflow.log_param("n_eval_queries", len(eval_df))
        mlflow.log_param("data_dir", str(DATA_DIR))
        mlflow.log_param("evaluation_dataset", str(EVALUATION_DATASET))

        for index, combination in enumerate(combinations, 1):
            params = dict(zip(keys, combination))

            run_name = (
                f"chunk{params['chunk_size']}"
                f"_overlap{params['chunk_overlap']}"
                f"_k{params['n_results']}"
            )

            logger.info("[%d/%d] Testing %s", index, len(combinations), run_name)

            # ---------------------------------------------------------
            # FIXED TEMP DIRECTORY HANDLING FOR WINDOWS
            # ---------------------------------------------------------
            tmp_dir = f"./temp_chroma_{index}"
            os.makedirs(tmp_dir, exist_ok=True)

            start_time = time.perf_counter()

            documents = process_data_folder_for_experiment(
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"]
            )

            collection = build_temp_collection(
                documents=documents,
                tmp_dir=tmp_dir,
                model=model
            )

            ingest_secs = round(time.perf_counter() - start_time, 2)

            metrics = evaluate_configuration(
                collection=collection,
                model=model,
                eval_df=eval_df,
                n_results=params["n_results"]
            )

            metrics["n_chunks"] = len(documents)
            metrics["ingest_secs"] = ingest_secs

            # ---------------------------------------------------------
            # RELEASE CHROMADB FILE HANDLES
            # ---------------------------------------------------------
            try:
                del collection
            except Exception:
                pass

            gc.collect()
            time.sleep(2)

            with mlflow.start_run(
                run_name=run_name,
                nested=True
            ):
                mlflow.set_tag("parent_run_id", parent_run.info.run_id)
                mlflow.set_tag("run_type", "rag_config")

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

            logger.info(
                "Result: hit_rate=%.2f | rag_score=%.4f | poor=%.2f%% | chunks=%d",
                metrics["hit_rate"],
                metrics["mean_rag_score"],
                metrics["pct_poor"],
                metrics["n_chunks"]
            )

            ranking_score = metrics["hit_rate"] + (metrics["mean_rag_score"] * 100)

            if ranking_score > best_score:
                best_score = ranking_score
                best_config = {
                    **params,
                    **metrics
                }

        if best_config:
            for key, value in best_config.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"best_{key}", value)
                else:
                    mlflow.log_param(f"best_{key}", value)

            mlflow.set_tag("best_config", str(best_config))

    logger.info("Best configuration: %s", best_config)
    logger.info("Experiment finished successfully.")

if __name__ == "__main__":
    run_experiment()