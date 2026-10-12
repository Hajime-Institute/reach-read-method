#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reach-Read Method の日本語データ（有価証券報告書 2,551 通と、それを検索させる設問 406 問）を取得する。

EDINET-Bench（Sakana AI、公共データ利用規約 第1.0版）を版を固定して Hugging Face から取り、
論文の日本語の索引と同じ手順で 2,551 通を組み直す。
設問ファイルを渡すと、日本語の設問のうちこの 2,551 通を検索させる 406 問を取り出し、
正解の数値が印字された位置を書き出した本文と照らして確かめる。

    python get_ja_data.py --out ja_data --questions benchmark.json --chunks

使い方と出力の説明は README.md。
"""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = "SakanaAI/EDINET-Bench"
REVISION = "fdd5db0f21567ebc5ff55a169312b6ca21735376"  # 論文の索引を作った版
# 同じ書類が split をまたいで 34 通入っている。先に読んだほうを残すので、読む順も固定する
PARTS = [("fraud_detection", "train"), ("fraud_detection", "test"),
         ("earnings_forecast", "train"), ("earnings_forecast", "test"),
         ("industry_prediction", "train")]
PERIOD_JA = {"CurrentYear": "当期", "FilingDate": "提出日時点"}
CHUNK, OVERLAP, MIN_SECTION = 800, 150, 80  # 論文の索引と同じ区切り方
EXPECT = {"docs": 2551, "companies": 1972, "duplicates": 34, "chunks": 170029,
          "questions": 406, "spans": 1160}


def fetch(snapshot, revision):
    """EDINET-Bench の置き場所を返す。--snapshot が無ければ Hugging Face から取る。"""
    if snapshot:
        return Path(snapshot)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub が入っていません。先に pip install -r requirements.txt を実行してください。")
    print(f"取得: {REPO}@{revision[:7]}（約 112 MB。2 回目からはキャッシュを使う）", flush=True)
    return Path(snapshot_download(REPO, repo_type="dataset", revision=revision,
                                  allow_patterns=["*/*.parquet", "LICENSE*", "README.md"]))


def fiscal_period(meta):
    m = re.match(r"(\d{4})-(\d{2})-\d{2}", (meta.get("当事業年度終了日") or "").strip())
    return f"{int(m.group(1))}年{int(m.group(2))}月期" if m else None


def load_docs(snap):
    """5 つの parquet から書類を組む。落とし方は論文の索引と同じ。

    XBRL の数値（bs・pl・cf の欄）は読まない。設問の正解はこの数値から作ったので、
    RAG に入れると答えを渡すことになる。
    """
    import pyarrow.parquet as pq
    docs, seen, dup, drop = [], set(), 0, Counter()
    for split, part in PARTS:
        path = snap / split / f"{part}-00000-of-00001.parquet"
        if not path.exists():
            sys.exit(f"{path} がありません。--snapshot の場所か、取得が終わったかを確かめてください。")
        cols = [c for c in ("doc_id", "edinet_code", "meta", "text", "industry")
                if c in pq.read_schema(path).names]
        for r in pq.read_table(path, columns=cols).to_pylist():
            if r["doc_id"] in seen:
                dup += 1
                continue
            seen.add(r["doc_id"])
            try:
                meta = json.loads(r["meta"])
            except Exception:
                drop["meta"] += 1
                continue
            fp = fiscal_period(meta)
            if fp is None:
                drop["fiscal_year_end"] += 1
                continue
            try:
                sections = json.loads(r["text"])
            except Exception:
                drop["text"] += 1
                continue
            docs.append({
                "doc_id": r["doc_id"], "edinet_code": r["edinet_code"],
                "company": meta.get("会社名", ""), "sec_code": meta.get("証券コード", ""),
                "fiscal_period": fp, "fiscal_year_end": meta.get("当事業年度終了日", ""),
                "gaap": meta.get("会計基準", ""), "consolidated": meta.get("連結決算の有無", ""),
                "amended": meta.get("修正の有無", ""), "industry": r.get("industry"),
                "source": f"{split}/{part}", "sections": sections,
            })
    return docs, dup, drop


def section_items(sections):
    """(節名, 期, 本文) を元のデータの順に出す。"""
    for name, v in sections.items():
        if isinstance(v, dict):
            for period, body in v.items():
                if isinstance(body, str):
                    yield name, period, body
        elif isinstance(v, str):
            yield name, "", v


def chunk_spans(n):
    """長さ n の本文を CHUNK 字ずつ OVERLAP 字重ねて区切る（論文の索引と同じ）。"""
    i = 0
    while i < n:
        yield i, min(i + CHUNK, n)
        if i + CHUNK >= n:
            break
        i += CHUNK - OVERLAP


def render(doc):
    """書類を 1 つのテキストにする。節ごとに本文の開始位置（文字数）を記録し、論文の区切り方のチャンクも返す。"""
    head = f"# {doc['company']} {doc['fiscal_period']} 有価証券報告書（{doc['doc_id']}）\n"
    parts, pos, secs, chunks = [head], len(head), [], []
    for name, period, body in section_items(doc["sections"]):
        if not body:
            continue
        label = f"\n## {name}（{PERIOD_JA.get(period, period)}）\n\n" if period else f"\n## {name}\n\n"
        parts.append(label)
        pos += len(label)
        secs.append({"section": name, "period": period, "offset": pos, "length": len(body)})
        if len(body) >= MIN_SECTION:
            sec = f"{name}/{period}" if period else name
            for a, b in chunk_spans(len(body)):
                chunks.append({"chunk_id": f"{doc['doc_id']}#{len(chunks)}", "doc_id": doc["doc_id"],
                               "section": sec, "start": pos + a, "end": pos + b})
        parts.append(body)
        pos += len(body)
    parts.append("\n")
    return "".join(parts), secs, chunks


def load_questions(path):
    """日本語の設問のうち、2,551 通を検索させるものを取り出す。

    設問ファイルは、欄の名前が id・documents・grading の形と
    task_id・docs・checks の形のどちらでも読む。
    """
    data = json.load(open(path, encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("tasks") or data.get("questions") or [])
    out = []
    for x in items:
        spec = x.get("documents") or x.get("docs") or {}
        if x.get("lang") == "ja" and str(spec.get("pool", "")).startswith("edinet"):
            out.append(x)
    return out


def locate(questions, index, out):
    """正解の印字の位置を、書き出したテキストファイルの位置に直して照らす。返すのは (一致, 照らした数)。"""
    texts, ok, n = {}, 0, 0
    with open(out / "questions_ja.jsonl", "w", encoding="utf-8") as f:
        for x in questions:
            gold = x.get("gold") or {}
            locs = []
            for ev in gold.get("evidence") or []:
                if not isinstance(ev, dict) or "char_span" not in ev:
                    continue
                n += 1
                doc_id = ev["doc_id"]
                sec = index.get((doc_id, ev["section"], ev.get("section_period") or ""))
                if sec is None:
                    print(f"  NG {x.get('id') or x.get('task_id')}: {doc_id} に節 {ev['section']} がない")
                    continue
                if doc_id not in texts:
                    # newline="" で読む。改行の変換が入ると位置がずれる
                    texts[doc_id] = open(out / "text" / f"{doc_id}.txt", encoding="utf-8", newline="").read()
                a, b = sec["offset"] + ev["char_span"][0], sec["offset"] + ev["char_span"][1]
                found = texts[doc_id][a:b]
                if found == ev["printed"]:
                    ok += 1
                else:
                    print(f"  NG {x.get('id') or x.get('task_id')}: 位置の文字列 {found!r} ≠ 印字 {ev['printed']!r}")
                locs.append({"doc_id": doc_id, "file": f"text/{doc_id}.txt", "span": [a, b],
                             "printed": ev["printed"], "unit": ev.get("unit"), "account": ev.get("account")})
            f.write(json.dumps({
                "id": x.get("id") or x.get("task_id"), "family": x.get("family"),
                "question": x.get("question"), "gold": gold,
                "grading": x.get("grading") or x.get("checks"), "gold_locations": locs,
            }, ensure_ascii=False) + "\n")
    return ok, n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="ja_data", help="書き出す場所（既定 ja_data）")
    ap.add_argument("--questions", help="設問ファイル（benchmark.json か suite_unified.json）")
    ap.add_argument("--chunks", action="store_true", help="論文の索引と同じ区切り方のチャンクも書き出す")
    ap.add_argument("--snapshot", help="取得済みの EDINET-Bench の場所。指定するとネットに出ない")
    ap.add_argument("--revision", default=REVISION, help="EDINET-Bench の版（既定は論文の版）")
    args = ap.parse_args()

    snap = fetch(args.snapshot, args.revision)
    out = Path(args.out)
    (out / "text").mkdir(parents=True, exist_ok=True)

    print("書類を組み直しています…", flush=True)
    docs, dup, drop = load_docs(snap)
    index, n_chunks = {}, 0
    chunk_f = open(out / "chunks.jsonl", "w", encoding="utf-8") if args.chunks else None
    with open(out / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            text, secs, chunks = render(d)
            with open(out / "text" / f"{d['doc_id']}.txt", "w", encoding="utf-8", newline="") as t:
                t.write(text)
            rec = {k: v for k, v in d.items() if k != "sections"}
            rec.update(file=f"text/{d['doc_id']}.txt", sections=secs)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for s in secs:
                index[(d["doc_id"], s["section"], s["period"])] = s
            n_chunks += len(chunks)
            if chunk_f:
                for c in chunks:
                    chunk_f.write(json.dumps(c, ensure_ascii=False) + "\n")
    if chunk_f:
        chunk_f.close()

    got = {"docs": len(docs), "companies": len({d["edinet_code"] for d in docs}), "duplicates": dup}
    if args.chunks:
        got["chunks"] = n_chunks
    if args.questions:
        qs = load_questions(args.questions)
        got["questions"] = len(qs)
        got["spans"], n_spans = locate(qs, index, out)
        got["spans_checked"] = n_spans

    for name in ("LICENSE", "LICENSE.pdf"):
        if (snap / name).exists():
            shutil.copy(snap / name, out / f"EDINET-Bench_{name}")
    json.dump({
        "source": f"https://huggingface.co/datasets/{REPO}", "revision": args.revision,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": {f"{s}/{p}-00000-of-00001.parquet": sha256(snap / s / f"{p}-00000-of-00001.parquet")
                  for s, p in PARTS},
        "counts": got, "dropped": dict(drop),
        "chunking": {"chars": CHUNK, "overlap": OVERLAP, "min_section_chars": MIN_SECTION},
        "license": "公共データ利用規約 第1.0版（PDL1.0）",
        "attribution": "EDINET-Bench（Sakana AI）を加工して作成。元データは金融庁 EDINET の有価証券報告書",
    }, open(out / "MANIFEST.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    labels = {"docs": "書類", "companies": "会社", "duplicates": "重複で落とした書類",
              "chunks": "チャンク", "questions": "設問", "spans": "本文と一致した正解の位置"}
    print("\n検査（期待する数は既定の版のもの）:")
    bad = 0
    for k, label in labels.items():
        if k not in got:
            continue
        good = got[k] == EXPECT[k] and (k != "spans" or got["spans"] == got["spans_checked"])
        bad += not good
        print(f"  {'OK' if good else 'NG'}  {label}: {got[k]:,}（期待 {EXPECT[k]:,}）")
    if drop:
        print(f"  読めずに落とした書類: {dict(drop)}")
    print(f"\n書き出し先: {out.resolve()}")
    if bad and args.revision == REVISION:
        sys.exit("NG があります。論文と同じ書類集合になっていません。")


if __name__ == "__main__":
    main()
