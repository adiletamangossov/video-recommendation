"""Чтение выгрузки каталога и приведение к единой таблице брендов.

Поддержан вариант А: единый файл каталога по контракту заказчика
в csv, parquet, json или jsonl. Вариант Б (сырые таблицы БД каталога) не реализован,
в выданных выгрузках его нет, см. load_from_db_tables.
"""

import difflib
import json
import os

import numpy as np
import pandas as pd

from video_recsys.config import STANDARD_AGES

TRUE_VALUES = {"ИСТИНА", "TRUE", "True", "true", "1", "t", "yes"}

TEXT_FALLBACK = {
    "title": ["title_ru", "title_native", "title_en"],
    "description": ["description_ru", "description_native", "description_en"],
}


def detect_separator(path, options=(";", "$", ",", "\t")):
    """Разделитель по строке заголовка. В выгрузках 16.09 был ;, в выгрузках 04.11 стал $."""
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
    return max(options, key=header.count)


def category_group(categories):
    """video_category -> группа кандидатов из CONFIG["category_groups"].

    Категория, которой нет в конфиге, остаётся отдельной группой.
    """
    from video_recsys.config import CONFIG
    groups = CONFIG["category_groups"]
    return categories.map(lambda c: groups.get(c, c))


def read_catalog(path):
    """Читает файл выгрузки как строки, без интерпретации пустых значений."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, sep=detect_separator(path), encoding="utf-8-sig",
                           dtype=str, keep_default_na=False)
    if ext == ".parquet":
        return pd.read_parquet(path).astype(str).fillna("")
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, dtype=str).fillna("")
    if ext == ".json":
        return pd.read_json(path, dtype=str).fillna("")
    raise ValueError(f"неизвестный формат выгрузки: {ext}")


def load_from_db_tables(connection):
    """Вариант Б: сборка каталога из сырых таблиц БД каталога.

    Не реализовано: обе выданные выгрузки пришли в варианте А. Функция должна вернуть
    такой же DataFrame, как build_brand_table, то есть по строке на медиабренд с
    колонками content_id, video_category, title, text, genre_ids, main_genre_ids,
    person_ids, age, is_ready.
    """
    raise NotImplementedError(
        "вариант Б не реализован, выгрузки пришли единым файлом по контракту заказчика")


def parse_json_array(value):
    """Разбирает ячейку с JSON массивом в список словарей."""
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def pick_first_filled(row, fields):
    """Fallback ru -> native -> en."""
    for field in fields:
        value = str(row.get(field, "") or "").strip()
        if value:
            return value
    return ""


def snap_to_standard_age(years):
    """Приводит возраст к стандартной шкале 0, 6, 12, 16, 18 ближайшим значением.

    В prod выгрузке встречаются значения вне шкалы (13 и 15, по одному бренду).
    При равном расстоянии до двух соседей берём меньшее.
    """
    return float(min(STANDARD_AGES, key=lambda standard: (abs(standard - years), standard)))


def parse_age(value, normalize=True):
    """Возраст в годах из age_restriction. Пустое и нечисловое дают nan."""
    text = str(value or "").strip().rstrip("+")
    if not text:
        return np.nan
    try:
        years = int(text)
    except ValueError:
        return np.nan
    return snap_to_standard_age(years) if normalize else float(years)


def build_brand_table(raw):
    """Единая таблица брендов: по строке на медиабренд, признаки уже разобраны."""
    brands = raw[raw["content_type"] == "MEDIABRAND"].copy()

    genres = brands["genres"].map(parse_json_array)
    actors = brands["actors"].map(parse_json_array)
    directors = brands["directors"].map(parse_json_array)

    table = pd.DataFrame({
        "content_id": brands["content_id"].str.strip(),
        "video_category": brands["video_category"].str.strip(),
        "title_ru": brands["title_ru"].str.strip(),
        "age_restriction": brands["age_restriction"].str.strip(),
        # Год нужен только для ревью дублей руками, в признаках v0 не участвует
        "release_year": brands["release_year"].str.strip(),
        "is_ready": brands["is_ready"].isin(TRUE_VALUES),
    })

    table["title"] = brands.apply(lambda r: pick_first_filled(r, TEXT_FALLBACK["title"]), axis=1).values
    table["description"] = brands.apply(
        lambda r: pick_first_filled(r, TEXT_FALLBACK["description"]), axis=1).values
    table["text"] = (table["title"] + " " + table["description"]).str.strip()

    table["genre_ids"] = [[g.get("genre_id") for g in gs if g.get("genre_id")] for gs in genres]
    table["main_genre_ids"] = [
        [g.get("genre_id") for g in gs if g.get("genre_id") and str(g.get("order")) == "1"]
        for gs in genres
    ]
    table["person_ids"] = [
        [p.get("person_id") for p in (a + d) if p.get("person_id")]
        for a, d in zip(actors, directors)
    ]
    table["group"] = category_group(table["video_category"])
    table["age"] = brands["age_restriction"].map(parse_age).values
    # Возраст как он пришёл, чтобы было видно, что именно нормализовали
    table["age_raw"] = brands["age_restriction"].map(
        lambda v: parse_age(v, normalize=False)).values

    return table.reset_index(drop=True)


def apply_overrides(table, overrides_path):
    """Применяет ручные правки каталога из json файла.

    Выгрузка это источник правды, поэтому её не меняем, а держим правки отдельным
    файлом. Пока поддержана одна операция, add_genre_ids: дописать жанр бренду.
    Жанр добавляется неосновным, то есть с весом other_weight, основной жанр
    бренда не меняется.

    Возвращает таблицу и список применённых правок для отчёта.
    """
    if not overrides_path or not os.path.exists(overrides_path):
        return table, []

    with open(overrides_path, encoding="utf-8") as f:
        overrides = json.load(f)

    by_id = table.set_index("content_id")
    applied = []
    for entry in overrides:
        content_id = entry["content_id"]
        if content_id not in by_id.index:
            applied.append({**entry, "статус": "бренда нет в выгрузке"})
            continue

        position = table.index[table["content_id"] == content_id][0]
        existing = list(table.at[position, "genre_ids"])
        added = [g for g in entry.get("add_genre_ids", []) if g not in existing]
        table.at[position, "genre_ids"] = existing + added
        applied.append({
            **entry,
            "статус": "применено" if added else "жанр уже был",
            "добавлено": len(added),
        })

    return table, applied


def describe_title_duplicates(table):
    """Группы брендов с одинаковым названием с пометкой, настоящий ли это дубль.

    Одинаковое название ещё не значит одну и ту же единицу контента: в каталоге есть
    ремейки и разные фильмы тёзки. Настоящим дублем считаем пару, у которой описания
    почти совпадают.
    """
    norm = table["title"].str.lower().str.strip()
    counts = norm.value_counts()
    dup_titles = set(counts[counts > 1].index) - {""}

    rows = []
    for title in sorted(dup_titles):
        group = table[norm == title]
        kept = group.iloc[0]
        for _, other in group.iloc[1:].iterrows():
            similarity = difflib.SequenceMatcher(
                None, kept["description"], other["description"]).ratio()
            rows.append({
                "title": kept["title"],
                "content_id_a": kept["content_id"],
                "year_a": kept["release_year"],
                "content_id_b": other["content_id"],
                "year_b": other["release_year"],
                "desc_similarity": round(float(similarity), 3),
            })
    return pd.DataFrame(rows)


def drop_title_duplicates(table, text_similarity_threshold=None):
    """Убирает бренды с повторяющимся названием.

    Решение владельца от 12.10.2026: убираем все одноимённые бренды. Из группы
    остаётся один: сначала готовый, потом с более длинным текстом.

    Если задан text_similarity_threshold, удаляются только настоящие дубли, то есть
    те, чьё описание похоже на оставленное не меньше порога. Разные фильмы тёзки
    тогда остаются в каталоге.

    Возвращает очищенную таблицу и таблицу удалённых строк для отчёта.
    """
    norm = table["title"].str.lower().str.strip()
    counts = norm.value_counts()
    dup_titles = set(counts[counts > 1].index) - {""}
    if not dup_titles:
        return table.reset_index(drop=True), table.iloc[0:0].copy()

    ranked = table.assign(_norm=norm, _len=table["text"].str.len())
    ranked = ranked.sort_values(["is_ready", "_len"], ascending=[False, False])
    keep_by_title = ranked[ranked["_norm"].isin(dup_titles)].groupby("_norm")["content_id"].first()
    keep_ids = set(keep_by_title)

    removed_mask = norm.isin(dup_titles) & ~table["content_id"].isin(keep_ids)

    descriptions = table.set_index("content_id")["description"]
    similarity = pd.Series(np.nan, index=table.index, dtype=float)
    for idx in table.index[removed_mask]:
        kept_id = keep_by_title[norm[idx]]
        similarity[idx] = difflib.SequenceMatcher(
            None, descriptions[kept_id], table.at[idx, "description"]).ratio()

    if text_similarity_threshold is not None:
        removed_mask = removed_mask & (similarity >= text_similarity_threshold)

    removed = table[removed_mask].copy()
    removed["desc_similarity"] = similarity[removed_mask].round(3)
    removed["kept_content_id"] = [keep_by_title[norm[idx]] for idx in removed.index]
    kept = table[~removed_mask].reset_index(drop=True)
    return kept, removed.reset_index(drop=True)


def load_brands(path, drop_duplicates=True, text_similarity_threshold=None,
                overrides_path=None):
    """Полный путь: файл выгрузки -> таблица брендов.

    Возвращает таблицу, удалённые дубли и список применённых ручных правок.
    """
    raw = read_catalog(path)
    table = build_brand_table(raw)
    table, applied = apply_overrides(table, overrides_path)
    if drop_duplicates:
        table, removed = drop_title_duplicates(table, text_similarity_threshold)
    else:
        removed = table.iloc[0:0].copy()
    return table, removed, applied


def build_episode_table(raw, table):
    """Выпуски шоу: по строке на VIDEO, привязанные к строке своего бренда в table.

    Из выпуска берём только текст. Жанры, персоны и возраст берутся с бренда шоу:
    жанры у выпусков в выгрузках совпадают с брендом, а возраст в prod пуст у 81%.
    Если у выпуска нет описания, текст берётся с бренда, иначе от выпуска остаётся
    одно название вида «10 серия».

    Возвращает таблицу выпусков и выпуски, чьего бренда нет в table.
    """
    videos = raw[raw["content_type"] == "VIDEO"]
    brand_position = pd.Series(table.index, index=table["content_id"])

    episodes = pd.DataFrame({
        "content_id": videos["content_id"].str.strip(),
        "mediabrand_id": videos["mediabrand_id"].str.strip(),
        "title_ru": videos["title_ru"].str.strip(),
        "age_restriction": videos["age_restriction"].str.strip(),
        "is_ready": videos["is_ready"].isin(TRUE_VALUES),
    })
    title = videos.apply(lambda r: pick_first_filled(r, TEXT_FALLBACK["title"]), axis=1).values
    description = videos.apply(
        lambda r: pick_first_filled(r, TEXT_FALLBACK["description"]), axis=1).values
    episodes["description"] = description

    missing_mask = ~episodes["mediabrand_id"].isin(brand_position.index)
    missing = episodes[missing_mask].reset_index(drop=True)
    episodes = episodes[~missing_mask].copy()
    title = title[~missing_mask.to_numpy()]

    episodes["brand_idx"] = brand_position[episodes["mediabrand_id"]].to_numpy()
    own_text = (pd.Series(title, index=episodes.index) + " " + episodes["description"]).str.strip()
    brand_text = table["text"].to_numpy()[episodes["brand_idx"]]
    has_description = episodes["description"] != ""
    episodes["text"] = np.where(has_description, own_text, brand_text)
    episodes["text_source"] = np.where(has_description, "выпуск", "бренд")

    return episodes.reset_index(drop=True), missing


def load_episodes(path, table):
    """Файл выгрузки -> таблица выпусков, привязанная к уже собранной таблице брендов."""
    return build_episode_table(read_catalog(path), table)


def age_normalization(table):
    """Бренды, у которых возраст не совпал со стандартной шкалой и был приведён."""
    changed = table[table["age"].notna() & (table["age"] != table["age_raw"])]
    return changed[["content_id", "title_ru", "age_raw", "age"]].copy()


def summarize(table):
    """Короткая сводка по таблице брендов для вывода в консоль."""
    total = len(table)
    return {
        "брендов": total,
        "по категориям": table["video_category"].value_counts().to_dict(),
        "is_ready": int(table["is_ready"].sum()),
        "пустое описание": int((table["description"].str.strip() == "").sum()),
        "пустой возраст": int(table["age"].isna().sum()),
        "возраст приведён к шкале": len(age_normalization(table)),
        "без жанров": int((table["genre_ids"].map(len) == 0).sum()),
        "без основного жанра": int((table["main_genre_ids"].map(len) == 0).sum()),
        "без персон": int((table["person_ids"].map(len) == 0).sum()),
    }
