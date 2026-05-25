"""
CityFoam RAG Retrieval Evaluation Utilities

This file is isolated from the main API.
It is used only by run_rag_experiment.py to score retrieval quality
for different RAG configurations.
"""

import math
from dataclasses import dataclass


DISTANCE_SCALE = 10.0
WARNING_THRESHOLD = 0.25
TOP_K_FOR_SCORING = 3


@dataclass
class RetrievalScore:
    gap_score: float
    decay_score: float
    is_poor: bool


def score_distances(distances: list[float]) -> RetrievalScore:
    """
    Calculates two retrieval quality scores from ChromaDB distances.

    gap_score:
        Measures how clearly the best retrieved chunk is separated
        from the worst retrieved chunk.

    decay_score:
        Converts the average top distances into a smooth score.
    """
    if not distances:
        return RetrievalScore(
            gap_score=0.0,
            decay_score=0.0,
            is_poor=True
        )

    sorted_distances = sorted(distances)
    best_dist = sorted_distances[0]
    worst_dist = sorted_distances[-1]
    top_distances = sorted_distances[:TOP_K_FOR_SCORING]

    if worst_dist == 0:
        gap_score = 1.0
    else:
        gap_score = 1.0 - (best_dist / worst_dist)

    mean_top_distance = sum(top_distances) / len(top_distances)
    decay_score = math.exp(-mean_top_distance / DISTANCE_SCALE)

    gap_score = round(float(gap_score), 4)
    decay_score = round(float(decay_score), 4)

    return RetrievalScore(
        gap_score=gap_score,
        decay_score=decay_score,
        is_poor=gap_score < WARNING_THRESHOLD
    )