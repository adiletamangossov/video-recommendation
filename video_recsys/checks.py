"""Автоматические проверки выдачи перед сдачей."""

import numpy as np
import pandas as pd


def check_coverage(recs, table, top_k):
    """1. Каждый бренд источник присутствует, кроме групп из одного бренда."""
    group_size = table["group"].map(table["group"].value_counts())
    eligible = table[group_size > 1]
    missing = int((~eligible["content_id"].isin(set(recs["content_id"]))).sum())
    short = int((recs.groupby("content_id").size() < top_k).sum())
    return {
        "проверка": "1. покрытие источников",
        "ок": missing == 0,
        "детали": f"без рекомендаций {missing} из {len(eligible)}, выдач короче {top_k}: {short}",
    }


def check_integrity(recs):
    """2. Нет self рекомендаций, нет дублей пары, rank без пропусков."""
    self_recs = int((recs["content_id"] == recs["similar_content_id"]).sum())
    dupes = int(recs.duplicated(["content_id", "similar_content_id"]).sum())

    bad_rank = 0
    for _, group in recs.groupby("content_id", sort=False):
        ranks = sorted(group["rank"])
        if ranks != list(range(1, len(ranks) + 1)):
            bad_rank += 1

    ok = self_recs == 0 and dupes == 0 and bad_rank == 0
    return {
        "проверка": "2. целостность выдачи",
        "ок": ok,
        "детали": f"self {self_recs}, дублей пар {dupes}, источников с битым rank {bad_rank}",
    }


def check_group_match(recs, table):
    """3. Группа категорий источника и кандидата совпадает: фильмы и сериалы вместе, шоу отдельно."""
    group = table.set_index("content_id")["group"]
    mismatch = int((recs["content_id"].map(group) != recs["similar_content_id"].map(group)).sum())
    return {
        "проверка": "3. совпадение группы категорий",
        "ок": mismatch == 0,
        "детали": f"строк с разной группой {mismatch} из {len(recs)}",
    }


def check_hubness(recs, limit, name="4. хабовость", groups=None, sample_size=50, seed=42):
    """4. На 50 случайных источниках максимум вхождений кандидата в топ 5 не выше limit.

    groups это Series content_id -> группа. Если задан, из каждой группы в выборку
    идёт не больше одного источника. Нужно для выпусков: выпуски одного шоу имеют
    почти одинаковый топ, и без этого проверка считает одно шоу несколько раз.
    """
    sources = recs["content_id"].unique()
    if groups is not None:
        rng = np.random.default_rng(seed)
        by_group = pd.Series(sources).groupby(groups.reindex(sources).to_numpy())
        sources = np.array([rng.choice(members.to_numpy()) for _, members in by_group])
    # sample падал, когда объектов меньше размера выборки
    take = min(sample_size, len(sources))
    rng = np.random.default_rng(seed)
    sample = rng.choice(sources, size=take, replace=False)

    top5 = recs[recs["content_id"].isin(sample) & (recs["rank"] <= 5)]
    counts = top5["similar_content_id"].value_counts()
    worst = int(counts.iloc[0]) if len(counts) else 0

    unit = "шоу" if groups is not None else "источников"
    return {
        "проверка": name,
        "ок": worst <= limit,
        "детали": (f"{unit} в выборке {take}, максимум вхождений кандидата в топ 5 "
                   f"{worst}, предел {limit}"),
    }


def check_score_distribution(recs):
    """6. Распределение score. Порог не задан, цифры идут в отчёт."""
    scores = recs["score"]
    return {
        "проверка": "6. распределение score",
        "ок": True,
        "детали": (f"mean {scores.mean():.4f}, медиана {scores.median():.4f}, "
                   f"min {scores.min():.4f}, max {scores.max():.4f}, "
                   f"отрицательных {100.0 * (scores < 0).mean():.1f}%"),
    }


def run_all(recs, table, config):
    """Все автоматические проверки. Ручное ревью см. sample_for_eyes."""
    shows = table.loc[table["video_category"] == "video_show", "content_id"]
    results = [
        check_coverage(recs, table, config["top_k"]),
        check_integrity(recs),
        check_group_match(recs, table),
        check_hubness(recs, config["hub_top5_limit"]),
        check_hubness(recs[recs["content_id"].isin(shows)],
                      config["hub_top5_limit_video_show"], name="4. хабовость, только шоу"),
        check_score_distribution(recs),
    ]
    return pd.DataFrame(results)


def sample_for_eyes(recs, table, per_category=4, top_n=5, seed=7):
    """Выборка для ручного ревью: по несколько источников каждой категории с их топом."""
    rng = np.random.default_rng(seed)
    titles = table.set_index("content_id")["title_ru"]
    categories = table.set_index("content_id")["video_category"]

    lines = []
    for category in sorted(table["video_category"].unique()):
        pool = [cid for cid in recs["content_id"].unique() if categories.get(cid) == category]
        if not pool:
            continue
        take = min(per_category, len(pool))
        for content_id in rng.choice(pool, size=take, replace=False):
            top = recs[(recs["content_id"] == content_id) & (recs["rank"] <= top_n)]
            lines.append(f"{category}: {titles.get(content_id, content_id)}")
            for _, row in top.sort_values("rank").iterrows():
                lines.append(f"    {int(row['rank'])}. {titles.get(row['similar_content_id'], '?')}"
                             f"  score {row['score']:.3f}")
            lines.append("")
    return "\n".join(lines)


def run_episodes(recs, episodes, table, config):
    """Проверки выдачи по выпускам VIDEO -> VIDEO: покрытие, целостность, группа, хабовость, score и доля своего шоу."""
    show_of = episodes.set_index("content_id")["mediabrand_id"]
    group = table.set_index("content_id")["group"]
    ep_group = episodes["mediabrand_id"].map(group)
    eligible = episodes[ep_group.map(ep_group.value_counts()) > 1]
    missing = int((~eligible["content_id"].isin(set(recs["content_id"]))).sum())

    known = set(episodes["content_id"])
    unknown = int((~recs["similar_content_id"].isin(known)).sum())
    self_recs = int((recs["content_id"] == recs["similar_content_id"]).sum())
    dupes = int(recs.duplicated(["content_id", "similar_content_id"]).sum())
    bad_rank = sum(
        sorted(group_["rank"]) != list(range(1, len(group_) + 1))
        for _, group_ in recs.groupby("content_id", sort=False))
    short = int((recs.groupby("content_id").size() < config["top_k"]).sum())

    src_show = recs["content_id"].map(show_of)
    cand_show = recs["similar_content_id"].map(show_of)
    mismatch = int((src_show.map(group) != cand_show.map(group)).sum())
    same_show = src_show == cand_show

    results = [
        {"проверка": "1. покрытие выпусков", "ок": missing == 0,
         "детали": f"без рекомендаций {missing} из {len(eligible)}, "
                   f"выдач короче {config['top_k']}: {short}"},
        {"проверка": "2. целостность выдачи",
         "ок": self_recs == 0 and dupes == 0 and bad_rank == 0 and unknown == 0,
         "детали": f"self {self_recs}, дублей пар {dupes}, источников с битым rank {bad_rank}, "
                   f"кандидат не выпуск из каталога {unknown}"},
        {"проверка": "3. совпадение группы категорий", "ок": mismatch == 0,
         "детали": f"строк с разной группой {mismatch} из {len(recs)}"},
        check_hubness(recs, config["hub_top5_limit_video_show"], groups=show_of),
        check_score_distribution(recs),
        {"проверка": "7. выпуски своего шоу",
         "ок": int(same_show.groupby(recs["content_id"]).sum().max()) <= config["episode_same_show_max"],
         "детали": f"предел {config['episode_same_show_max']}, максимум в выдаче "
                   f"{int(same_show.groupby(recs['content_id']).sum().max())}, "
                   f"в топ 20 {100.0 * same_show.mean():.1f}% строк, "
                   f"в топ 5 {100.0 * same_show[recs['rank'] <= 5].mean():.1f}%, "
                   f"выдач только из своего шоу "
                   f"{int(same_show.groupby(recs['content_id']).all().sum())} "
                   f"из {recs['content_id'].nunique()}"},
    ]
    return pd.DataFrame(results)
