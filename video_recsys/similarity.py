"""Матрица похожести, фильтр группы категорий, штрафы и топ K.

Считаем по группам (фильмы с сериалами, шоу отдельно, CONFIG["category_groups"]):
фильтр тогда бесплатный, а памяти нужно меньше, чем на полную матрицу каталога.
"""

import numpy as np
import pandas as pd

from video_recsys.config import CONFIG


def raw_scores(features, src_idx, cand_idx, weights):
    """S до штрафов: взвешенная сумма трёх косинусов."""
    text = features.text[src_idx] @ features.text[cand_idx].T
    genres = (features.genres[src_idx] @ features.genres[cand_idx].T).toarray()
    persons = (features.persons[src_idx] @ features.persons[cand_idx].T).toarray()
    return (weights["text"] * text
            + weights["genres"] * genres
            + weights["persons"] * persons).astype(np.float32)


def self_positions(src_idx, cand_idx):
    """Для каждого столбца кандидата номер строки того же бренда, если он есть."""
    row_of = {brand_idx: row for row, brand_idx in enumerate(src_idx)}
    return {col: row_of[brand_idx]
            for col, brand_idx in enumerate(cand_idx)
            if brand_idx in row_of}


def hub_scores(scores, self_pos):
    """hub_j это среднее S по столбцу кандидата j. Сам с собой в среднее не идёт."""
    col_sum = scores.sum(axis=0, dtype=np.float64)
    counts = np.full(scores.shape[1], float(scores.shape[0]))
    for col, row in self_pos.items():
        col_sum[col] -= scores[row, col]
        counts[col] -= 1.0
    counts[counts == 0] = 1.0
    return col_sum / counts


def age_penalty(ages, src_idx, cand_idx, per_year):
    """Штраф per_year * |age_i - age_j|. Если возраст одного из двух пустой, штраф 0."""
    age_src = ages[src_idx]
    age_cand = ages[cand_idx]
    diff = np.abs(age_src[:, None] - age_cand[None, :])
    return per_year * np.nan_to_num(diff, nan=0.0)


def top_k_for_group(table, features, config, group, diagnostics):
    """Топ K кандидатов для каждого источника внутри одной группы категорий."""
    group_mask = (table["group"] == group).to_numpy()
    src_idx = np.flatnonzero(group_mask)
    cand_idx = src_idx

    if len(src_idx) == 0 or len(cand_idx) == 0:
        diagnostics.append({"group": group, "sources": len(src_idx),
                            "candidates": len(cand_idx), "note": "кандидатов нет"})
        return pd.DataFrame(columns=["content_id", "similar_content_id", "rank", "score"])

    ages = table["age"].to_numpy(dtype=np.float64)

    scores = raw_scores(features, src_idx, cand_idx, config["weights"])
    mean_before = float(scores.mean())

    self_pos = self_positions(src_idx, cand_idx)
    hub = hub_scores(scores, self_pos)
    scores -= (config["hub_penalty"] * hub[None, :]).astype(np.float32)
    scores -= age_penalty(ages, src_idx, cand_idx, config["age_penalty_per_year"]).astype(np.float32)

    # Сам себе кандидатом быть не может
    for col, row in self_pos.items():
        scores[row, col] = -np.inf

    k = min(config["top_k"], len(cand_idx) - (1 if self_pos else 0))
    k = max(k, 0)

    content_ids = table["content_id"].to_numpy()
    rows_out = []
    for row, brand_idx in enumerate(src_idx):
        if k == 0:
            continue
        row_scores = scores[row]
        finite = np.flatnonzero(np.isfinite(row_scores))
        if finite.size == 0:
            continue
        take = min(k, finite.size)
        best = finite[np.argpartition(-row_scores[finite], take - 1)[:take]]
        best = best[np.argsort(-row_scores[best], kind="stable")]
        for rank, col in enumerate(best, start=1):
            rows_out.append((content_ids[brand_idx], content_ids[cand_idx[col]],
                             rank, float(row_scores[col])))

    result = pd.DataFrame(rows_out, columns=["content_id", "similar_content_id", "rank", "score"])

    finite_scores = scores[np.isfinite(scores)]
    diagnostics.append({
        "group": group,
        "sources": len(src_idx),
        "candidates": len(cand_idx),
        "k": k,
        "mean_S_before_penalties": round(mean_before, 4),
        "mean_S_after_penalties": round(float(finite_scores.mean()), 4),
        "mean_top_score": round(float(result["score"].mean()), 4) if len(result) else None,
        "hub_mean": round(float(hub.mean()), 4),
        "hub_max": round(float(hub.max()), 4),
    })
    return result


def build_recommendations(table, features, config, env):
    """Топ K по всем группам. Возвращает рекомендации и диагностику по группам."""
    diagnostics = []
    parts = [
        top_k_for_group(table, features, config, group, diagnostics)
        for group in sorted(table["group"].unique())
    ]
    recs = pd.concat(parts, ignore_index=True)
    return recs, pd.DataFrame(diagnostics)


def to_contract(recs):
    """Формат результата по контракту заказчика."""
    out = recs.copy()
    out["content_type"] = "MEDIABRAND"
    out["similar_content_type"] = "MEDIABRAND"
    out["score"] = out["score"].round(CONFIG["score_decimals"])
    return out[["content_type", "content_id", "similar_content_type",
                "similar_content_id", "rank", "score"]]


def to_review(recs, table):
    """Та же выдача плюс названия, категория и возраст обоих для ревью руками.

    age_restriction идёт как в выгрузке, age_used это возраст, по которому реально
    считался штраф, то есть после приведения к стандартной шкале.
    """
    cols = ["content_id", "title_ru", "video_category", "age_restriction", "age"]
    fields = table[cols].copy()
    fields["age"] = fields["age"].astype("Int64")

    source = fields.rename(columns={
        "title_ru": "source_title_ru",
        "video_category": "source_video_category",
        "age_restriction": "source_age_restriction",
        "age": "source_age_used",
    })
    candidate = fields.rename(columns={
        "content_id": "similar_content_id",
        "title_ru": "similar_title_ru",
        "video_category": "similar_video_category",
        "age_restriction": "similar_age_restriction",
        "age": "similar_age_used",
    })
    out = to_contract(recs).merge(source, on="content_id", how="left")
    return out.merge(candidate, on="similar_content_id", how="left")
