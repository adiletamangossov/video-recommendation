"""Прогон v0 на одной выгрузке.

Пример: python -m video_recsys.run --env dev --input <путь к csv> --out out
"""

import argparse
import os
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_recsys import checks, episodes, features, load, similarity
from video_recsys.config import CONFIG


def print_block(title, body):
    print(f"\n{title}")
    print(body)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["prod", "dev"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="out")
    parser.add_argument("--overrides", default=None,
                        help="json с ручными правками каталога. По умолчанию правки НЕ "
                             "применяются: выгрузка это источник правды, и результат "
                             "должен воспроизводиться из неё без наших файлов")
    parser.add_argument("--encoding", default=None,
                        help="кодировка csv на выходе, по умолчанию из config.py "
                             "(utf-8 без BOM). utf-8-sig вернёт BOM для Excel")
    parser.add_argument("--eyes", action="store_true",
                        help="напечатать выборку для ручного ревью")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    encoding = args.encoding or CONFIG["output_encoding"]

    table, removed, applied = load.load_brands(
        args.input,
        CONFIG["drop_title_duplicates"],
        CONFIG["title_duplicate_text_similarity"],
        args.overrides,
    )
    print_block(f"[{args.env}] таблица брендов",
                "\n".join(f"  {k}: {v}" for k, v in load.summarize(table).items()))
    normalized = load.age_normalization(table)
    for _, row in normalized.iterrows():
        print(f"  возраст {row['age_raw']:.0f} -> {row['age']:.0f}: {row['title_ru']}")

    if applied:
        print(f"\nручные правки каталога из {args.overrides}")
        for entry in applied:
            names = ", ".join(entry.get("add_genre_names", entry.get("add_genre_ids", [])))
            print(f"  {entry['title_ru']}: жанры += {names} [{entry['статус']}]")
    else:
        print("  ручные правки каталога не применялись, расчёт строго по выгрузке")
    if len(removed):
        threshold = CONFIG["title_duplicate_text_similarity"]
        if threshold is None:
            print(f"  удалено одноимённых брендов: {len(removed)}, "
                  f"похожесть описаний не проверялась")
        else:
            print(f"  удалено дублей по названию: {len(removed)}, "
                  f"порог похожести описаний {threshold}, "
                  f"минимальная похожесть среди удалённых "
                  f"{removed['desc_similarity'].min():.3f}")

    feats = features.build_all(table, CONFIG)
    print_block(f"[{args.env}] признаки",
                "\n".join(f"  {k}: {v}" for k, v in feats.info.items()))

    recs, diagnostics = similarity.build_recommendations(table, feats, CONFIG, args.env)
    print_block(f"[{args.env}] диагностика по группам категорий", diagnostics.to_string(index=False))

    report = checks.run_all(recs, table, CONFIG)
    print_block(f"[{args.env}] проверки", report.to_string(index=False))

    if len(removed):
        removed_path = os.path.join(args.out, f"removed_title_duplicates_{args.env}.csv")
        removed[["content_id", "title_ru", "video_category", "release_year",
                 "kept_content_id", "desc_similarity"]].to_csv(
            removed_path, index=False, encoding=encoding)
        print(f"\nудалённые дубли: {removed_path}")

    # Выпуски шоу, правила заказчика от 21.11.2026: VIDEO -> VIDEO
    episode_table, orphans = load.load_episodes(args.input, table)
    print_block(f"[{args.env}] выпуски",
                f"  выпусков: {len(episode_table)}\n"
                f"  текст с бренда, нет описания: "
                f"{int((episode_table['text_source'] == 'бренд').sum())}\n"
                f"  бренда нет в таблице, пропущено: {len(orphans)}")
    episode_recs, episode_diag = episodes.top_k_for_episodes(
        episode_table, table, feats, CONFIG)
    print_block(f"[{args.env}] диагностика по выпускам", episode_diag.to_string(index=False))
    episode_report = checks.run_episodes(episode_recs, episode_table, table, CONFIG)
    print_block(f"[{args.env}] проверки по выпускам", episode_report.to_string(index=False))

    # Решение владельца от 21.10.2026: клиенту отдаём бренды и выпуски одним файлом.
    # С 21.11.2026 отдельные файлы по брендам и по выпускам не пишем, они его части
    combined_path = os.path.join(args.out, f"recs_{args.env}_with_episodes.csv")
    combined = pd.concat([similarity.to_contract(recs), episodes.to_contract(episode_recs)],
                         ignore_index=True)
    combined.to_csv(combined_path, index=False, encoding=encoding)
    print(f"\nзаписано: {combined_path} ({len(combined)} строк), бренды и выпуски вместе, "
          f"кодировка {encoding}")

    # Ревью к нему. xlsx, потому что csv в utf-8 Excel открывает с битой кириллицей, а
    # разделитель зависит от локали. Колонки обеих частей, у брендов колонки выпусков пустые
    combined_review_path = os.path.join(args.out, f"recs_{args.env}_with_episodes_review.xlsx")
    combined_review = pd.concat([similarity.to_review(recs, table),
                                 episodes.to_review(episode_recs, episode_table, table)],
                                ignore_index=True)
    combined_review.to_excel(combined_review_path, index=False)
    print(f"записано: {combined_review_path}")

    if args.eyes:
        print_block(f"[{args.env}] выборка для ручного ревью",
                    checks.sample_for_eyes(recs, table))

    return 0 if bool(report["ок"].all() and episode_report["ок"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
