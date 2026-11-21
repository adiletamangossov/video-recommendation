"""«Похожее» для выпусков шоу: для каждого выпуска топ K похожих выпусков (VIDEO -> VIDEO).

Правила заказчика от 21.11.2026: связи VIDEO -> MEDIABRAND больше нет. Источник и кандидаты это
выпуски одной группы категорий. Выпуски того же шоу рекомендовать можно, сам себя нельзя.

Признаки выпуска: текст свой (без описания берётся текст бренда шоу), жанры, персоны и
возраст от бренда шоу. Модель текста обучена только на брендах, поэтому выдача по
брендам от выпусков не зависит.

Хаб штраф кандидата это среднее S по его столбцу среди выпусков источников группы,
сам с собой в среднее не идёт. Формула та же, что у брендов.
"""

import numpy as np
import pandas as pd

from video_recsys.config import CONFIG
from video_recsys.features import transform_text
from video_recsys.similarity import age_penalty, hub_scores


def episode_raw_scores(episode_text, features, brand_idx, weights):
    """S до штрафов между всеми выпусками группы: текст от выпусков, остальное от брендов."""
    text = episode_text @ episode_text.T
    genres = (features.genres[brand_idx] @ features.genres[brand_idx].T).toarray()
    persons = (features.persons[brand_idx] @ features.persons[brand_idx].T).toarray()
    return (weights["text"] * text
            + weights["genres"] * genres
            + weights["persons"] * persons).astype(np.float32)


def top_n(row_scores, cols, n):
    """n лучших столбцов из cols по score, по убыванию. -inf (сам выпуск) не берётся."""
    cols = cols[np.isfinite(row_scores[cols])]
    n = min(n, cols.size)
    if n == 0:
        return cols[:0]
    best = cols[np.argpartition(-row_scores[cols], n - 1)[:n]]
    return best[np.argsort(-row_scores[best], kind="stable")]


def pick_with_show_cap(row_scores, same_show, take, same_show_max):
    """Топ take по score, в котором выпусков своего шоу не больше same_show_max.

    Это то же самое, что идти по убыванию score и пропускать выпуски своего шоу после
    same_show_max: берём лучших своих в пределе и лучших чужих, сливаем по score.
    Если чужих не хватает, выдача короче take, предел не нарушается.
    """
    own = top_n(row_scores, np.flatnonzero(same_show), same_show_max)
    other = top_n(row_scores, np.flatnonzero(~same_show), take)
    merged = np.concatenate([own, other])
    merged = merged[np.argsort(-row_scores[merged], kind="stable")]
    return merged[:take]


def top_k_for_episodes(episodes, table, features, config):
    """Топ K выпусков для каждого выпуска. Возвращает выдачу и диагностику."""
    groups = table["group"].to_numpy()
    ages = table["age"].to_numpy(dtype=np.float64)

    episode_text = transform_text(list(episodes["text"]), features.text_model)
    all_brand_idx = episodes["brand_idx"].to_numpy()
    episode_group = groups[all_brand_idx]
    all_ids = episodes["content_id"].to_numpy()

    rows_out, diagnostics = [], []
    for group in sorted(set(episode_group)):
        ep_rows = np.flatnonzero(episode_group == group)
        brand_idx = all_brand_idx[ep_rows]
        if len(ep_rows) < 2:
            diagnostics.append({"group": group, "sources": len(ep_rows),
                                "candidates": len(ep_rows), "note": "кандидатов нет"})
            continue

        scores = episode_raw_scores(episode_text[ep_rows], features, brand_idx,
                                    config["weights"])
        mean_before = float(scores.mean())

        diagonal = {i: i for i in range(len(ep_rows))}
        hub = hub_scores(scores, diagonal)
        scores -= (config["hub_penalty"] * hub[None, :]).astype(np.float32)
        scores -= age_penalty(ages, brand_idx, brand_idx,
                              config["age_penalty_per_year"]).astype(np.float32)

        # Сам себе кандидатом быть не может
        np.fill_diagonal(scores, -np.inf)

        ids = all_ids[ep_rows]
        top_scores = []
        take = min(config["top_k"], len(ep_rows) - 1)
        for row in range(len(ep_rows)):
            best = pick_with_show_cap(scores[row], brand_idx == brand_idx[row],
                                      take, config["episode_same_show_max"])
            row_scores = scores[row]
            for rank, col in enumerate(best, start=1):
                rows_out.append((ids[row], ids[col], rank, float(row_scores[col])))
                top_scores.append(float(row_scores[col]))

        finite_scores = scores[np.isfinite(scores)]
        diagnostics.append({
            "group": group,
            "sources": len(ep_rows),
            "candidates": len(ep_rows),
            "k": take,
            "mean_S_before_penalties": round(mean_before, 4),
            "mean_S_after_penalties": round(float(finite_scores.mean()), 4),
            "mean_top_score": round(float(np.mean(top_scores)), 4),
            "hub_mean": round(float(hub.mean()), 4),
            "hub_max": round(float(hub.max()), 4),
        })

    recs = pd.DataFrame(rows_out, columns=["content_id", "similar_content_id", "rank", "score"])
    return recs, pd.DataFrame(diagnostics)


def to_contract(recs):
    """Тот же формат, что у брендов, источник и кандидат VIDEO."""
    out = recs.copy()
    out["content_type"] = "VIDEO"
    out["similar_content_type"] = "VIDEO"
    out["score"] = out["score"].round(CONFIG["score_decimals"])
    return out[["content_type", "content_id", "similar_content_type",
                "similar_content_id", "rank", "score"]]


def to_review(recs, episodes, table):
    """Выдача плюс названия выпусков, их шоу, откуда взят текст, и пометка «то же шоу»."""
    brand = table[["content_id", "title_ru", "video_category", "age_restriction", "age"]].copy()
    brand["age"] = brand["age"].astype("Int64")
    info = episodes[["content_id", "title_ru", "mediabrand_id", "text_source"]].merge(
        brand.rename(columns={"content_id": "mediabrand_id", "title_ru": "show_title_ru"}),
        on="mediabrand_id", how="left")

    def side(prefix, key):
        cols = {c: f"{prefix}_{c}" for c in info.columns if c != "content_id"}
        cols["content_id"] = key
        return info.rename(columns=cols)

    out = to_contract(recs).merge(side("source", "content_id"), on="content_id", how="left")
    out = out.merge(side("similar", "similar_content_id"), on="similar_content_id", how="left")
    out["same_show"] = out["source_mediabrand_id"] == out["similar_mediabrand_id"]
    return out
