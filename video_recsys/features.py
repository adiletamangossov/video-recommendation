"""Признаки v0: текст, жанры, персоны. Все матрицы L2 нормализованы, поэтому
скалярное произведение строк это сразу косинус.

Признаки считаются один раз по всему каталогу, покатегорийной остаётся только
матрица похожести. Порядок строк во всех матрицах совпадает с таблицей брендов.
"""

from collections import namedtuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

Features = namedtuple("Features", ["text", "genres", "persons", "info", "text_model"])

# Обученные на брендах TF IDF и SVD, чтобы перевести в то же пространство тексты выпусков
TextModel = namedtuple("TextModel", ["vectorizer", "svd"])


def build_text(texts, params):
    """TF IDF на символьных n граммах, затем TruncatedSVD и L2."""
    vectorizer = TfidfVectorizer(
        analyzer=params["analyzer"],
        ngram_range=params["ngram_range"],
        min_df=params["min_df"],
    )
    tfidf = vectorizer.fit_transform(texts)

    # SVD не может дать компонент больше, чем есть признаков или объектов
    n_components = min(params["svd_components"], min(tfidf.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=params["random_state"])
    dense = svd.fit_transform(tfidf)

    info = {
        "tfidf_shape": tfidf.shape,
        "svd_components": n_components,
        "svd_explained": float(svd.explained_variance_ratio_.sum()),
    }
    return normalize(dense.astype(np.float32)), info, TextModel(vectorizer, svd)


def transform_text(texts, text_model):
    """Тексты, не участвовавшие в обучении, в пространство текстовых векторов брендов."""
    dense = text_model.svd.transform(text_model.vectorizer.transform(texts))
    return normalize(dense.astype(np.float32))


def build_multi_hot(id_lists, weights_lists=None):
    """Разреженная multi hot матрица по списку id у каждого объекта, затем L2."""
    vocabulary = {}
    rows, cols, values = [], [], []
    for row, ids in enumerate(id_lists):
        weights = weights_lists[row] if weights_lists is not None else [1.0] * len(ids)
        for item_id, weight in zip(ids, weights):
            col = vocabulary.setdefault(item_id, len(vocabulary))
            rows.append(row)
            cols.append(col)
            values.append(weight)
    matrix = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(id_lists), max(len(vocabulary), 1)),
        dtype=np.float32,
    )
    # Один id мог прийти дважды, суммируем дубли в одну ячейку
    matrix.sum_duplicates()
    return normalize(matrix), len(vocabulary)


def build_genres(table, params):
    """Основной жанр (order = 1) весит main_weight, остальные other_weight."""
    id_lists, weight_lists = [], []
    for genre_ids, main_ids in zip(table["genre_ids"], table["main_genre_ids"]):
        main = set(main_ids)
        id_lists.append(list(genre_ids))
        weight_lists.append([
            params["main_weight"] if genre_id in main else params["other_weight"]
            for genre_id in genre_ids
        ])
    matrix, size = build_multi_hot(id_lists, weight_lists)
    return matrix, {"genres_vocab": size}


def build_persons(table):
    """Актёры и режиссёры в одном векторе, вес 1."""
    matrix, size = build_multi_hot(list(table["person_ids"]))
    return matrix, {"persons_vocab": size}


def build_all(table, config):
    """Собирает все признаки до любых расчётов похожести.

    Порядок важен: сначала все признаки,
    только потом матрица S.
    """
    text, text_info, text_model = build_text(list(table["text"]), config["text"])
    genres, genres_info = build_genres(table, config["genres"])
    persons, persons_info = build_persons(table)

    info = {}
    info.update(text_info)
    info.update(genres_info)
    info.update(persons_info)
    info["empty_text_vectors"] = int((np.abs(text).sum(axis=1) == 0).sum())
    info["empty_genre_vectors"] = int((genres.getnnz(axis=1) == 0).sum())
    info["empty_person_vectors"] = int((persons.getnnz(axis=1) == 0).sum())

    return Features(text=text, genres=genres, persons=persons, info=info, text_model=text_model)
