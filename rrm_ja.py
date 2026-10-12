# -*- coding: utf-8 -*-
"""Reach-Read Method の日本語版（有価証券報告書の設問で使う照合と採点の規則）。

英語版（米国企業の年次報告書 10-K で使ったもの）と同じ考え方で、日本語の書き方に合わせた。
言語モデルも人手も使わない。

  Read  回答の値の欄を、正答・無回答・誤答の三つに分ける。
        相対誤差 0.5% 以内で、しかも回答に書かれた最後の桁で正解を丸めると一致するときだけ正答。
        英語版から足したのは、全角数字・△▲・「マイナス」・億・兆・千円・百万円・
        「2,458億96百万円」のような組み合わせ・日本語の「見つかりません」の類。
  Reach チャンクが対象企業の書類のもので、正解の数値が印字されていれば「正解の箇所」。
        4 桁以上の表記はそのまま数え、3 桁以下と丸めた表記（「2,459億円」など）は、
        すぐ前に項目名があるときだけ数える。日本語の表は行が改行で区切られていないので、
        英語版の「同じ行」の代わりに「前 80 字以内」を使う。

    python3 rrm_ja.py --selftest                  手で作った例で規則を確かめる
    python3 rrm_ja.py --check ja_data             get_ja_data.py の出力で、正解の位置がすべて Reach に数えられるかと、
                                                  Read が正しい書き方と間違った書き方を分けるかを確かめる
    python3 rrm_ja.py --score run.jsonl --data ja_data
                                                  RAG の実行記録から、六つのマスと「まず直す場所」を出す
"""
import re
import unicodedata

TOL = 0.005
SHIFTS = (1, 1e3, 1e6)  # 単位の書いていない回答は、円・千円・百万円のどれで読んでもよい（英語版と同じ寛容さ）
BIG = {"兆": 1e12, "億": 1e8, "万": 1e4}
TAIL = (("百万円", 1e6), ("千円", 1e3), ("万円", 1e4), ("億円", 1e8), ("兆円", 1e12), ("円", 1.0),
        ("百万", 1e6), ("千", 1e3),
        ("billion", 1e9), ("million", 1e6), ("thousand", 1e3), ("bn", 1e9), ("mn", 1e6))
LABEL_WINDOW = 80

NUM = re.compile(r"(△|▲|マイナス|-)?\s*(\d[\d,]*(?:\.\d+)?)")
# 数値の前に読み飛ばす、年度・日付・EDINET コード。「2015年3月期の値は…」の 2015 を答えと取り違えないため
CONTEXT = re.compile(
    r"(?:19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日|期|末)?)?(?:度)?"
    r"|(?:19|20)\d{2}-\d{1,2}(?:-\d{1,2})?"
    r"|(?:令和|平成|昭和)\s*(?:\d+|元)\s*年(?:\s*\d{1,2}\s*月(?:期|末)?)?(?:度)?"
    r"|\d{1,2}\s*月期|第\s*\d+\s*期|FY\s*\d{2,4}|\bE\d{5}\b")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NOANS_LEAD = re.compile(
    r"^\W*(?:不明|該当なし|該当する|記載なし|記載が(?:ない|ありません)|見つから|見当たら|特定でき|判断でき|回答でき|"
    r"お答えでき|算出でき|情報(?:が|は)?(?:ない|なし|ありません)|データ(?:が|は)?(?:ない|なし|ありません)|なし|"
    r"not\b|no\b|n/?a\b|none\b|unknown|unavailable|cannot|could not|unable|insufficient|undisclosed)", re.I)
NOANS_ANY = re.compile(
    r"見つかりません|見つからない|見つかりませんでした|見当たりません|見当たらない|"
    r"記載(?:が|は)?(?:ない|なし|ありません|されていない|されておりません|されていません)|"
    r"含まれていない|含まれておりません|含まれていません|特定できない|特定できません|判断できない|判断できません|"
    r"回答できない|回答できません|不明|該当なし|開示されていない|開示されていません|"
    r"情報(?:が|は)?(?:ない|ありません|不足)|提供されていない|提供されていません|確認できない|確認できません|"
    r"\bnot (?:found|disclosed|reported|provided|available|stated|included)\b|\bcannot\b|\bunable\b|\binsufficient\b",
    re.I)


def _norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    return s.replace("−", "-").replace("‐", "-").replace("–", "-")


def _precision(txt):
    """書かれた最後の桁の大きさ。「245,000」は千の位まで、「2458.96」は 0.01 まで書いたとみなす。"""
    digits = txt.replace(",", "")
    if "." in digits:
        return 10.0 ** (-len(digits.split(".")[1]))
    tz = len(digits) - len(digits.rstrip("0"))
    return 10.0 ** tz if digits.strip("0") else 1.0


def parse(value, unit=None):
    """回答の値を (円の値, 最後の桁の大きさ, 単位が書いてあるか) にする。読めなければ (None, None, False)。

    「2,458億96百万円」は 2,458億 と 96百万円 の和として読む。単位の欄が別にある回答は unit で渡す。
    """
    if isinstance(value, bool):
        return None, None, False
    if isinstance(value, (int, float)):
        mul = dict(TAIL).get(_norm(unit).strip(), 1.0) if unit else 1.0
        return float(value) * mul, mul, bool(unit)
    s = CONTEXT.sub(" ", _norm(value) + (_norm(unit) if unit else ""))
    m = NUM.search(s)
    if not m:
        return None, None, False
    sign = -1.0 if m.group(1) else 1.0
    total, has_unit, pos, num = 0.0, False, m.end(), m.group(2)
    while True:
        x, prec = float(num.replace(",", "")), _precision(num)
        big = re.match(r"\s*(兆|億|万)(?!円)", s[pos:])
        if not big:
            break
        total += x * BIG[big.group(1)]
        has_unit, pos = True, pos + big.end()
        nxt = re.match(r"\s*(\d[\d,]*(?:\.\d+)?)", s[pos:])
        if not nxt:  # 「1兆2,345億」のように億や兆で終わる
            return sign * total, prec * BIG[big.group(1)], True
        num, pos = nxt.group(1), pos + nxt.end()
    tail = s[pos:].strip().lower()
    mul = 1.0
    for word, mm in TAIL:
        if tail.startswith(word):
            mul, has_unit = mm, True
            break
    return sign * (total + x * mul), prec * mul, has_unit


def is_no_answer(value, unit=None):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    v = _norm(value)
    if parse(value, unit)[0] is None or "INSUFFICIENT" in v.upper() or "UNKNOWN" in v.upper():
        return True
    if NOANS_LEAD.search(v):
        return True
    rest = YEAR.sub(" ", CONTEXT.sub(" ", v))
    return bool(NOANS_ANY.search(v)) and not re.search(r"\d", rest)


def classify(value, gold, unit=None):
    """exact・rounded・near・wrong・novalue のどれか。英語版の classify と同じ段階。"""
    v, prec, has_unit = parse(value, unit)
    if v is None:
        return "novalue"
    if gold is None:
        return "wrong"
    gold = float(gold)
    if gold == 0:
        return "exact" if abs(v) < 1e-9 else "wrong"
    best = None
    for m in ((1,) if has_unit else SHIFTS):
        diff = abs(v * m - gold)
        if diff > abs(gold) * TOL:
            continue
        if diff <= max(1e-9 * abs(gold), 1e-9):
            return "exact"
        if diff <= prec * m / 2 * (1 + 1e-9):
            best = "rounded"
        elif best is None:
            best = "near"
    return best or "wrong"


def read(value, gold, unit=None):
    """正答（correct）・無回答（abstained）・誤答（wrong）。"""
    if is_no_answer(value, unit):
        return "abstained"
    return "correct" if gold is not None and classify(value, gold, unit) in ("exact", "rounded") else "wrong"


# ---------------------------------------------------------------- Reach
def _digits(s):
    return len(re.sub(r"\D", "", s))


def forms(value_yen):
    """正解の円の値が書類に印字されうる表記を、(そのまま数える表記, 項目名が要る表記) に分ける。"""
    av = abs(int(round(float(value_yen))))
    strong, weak = set(), set()

    def add(s):
        if re.fullmatch(r"(?:19|20)\d\d", s):
            return  # 年と同じ形は使わない
        (strong if _digits(s) >= 4 else weak).add(s)

    for div in (1, 1_000, 10_000, 1_000_000):  # 円・千円・万円・百万円で割り切れる表記
        if av % div == 0 and av // div >= 1:
            add(f"{av // div:,}")
    if av % 1_000_000 == 0 and av >= 100_000_000:  # 「2,458億96百万」「1兆2,345億67百万」「75億4千5百万」
        cho, r = divmod(av // 1_000_000, 1_000_000)
        oku, mil = divmod(r, 100)
        head = (f"{cho:,}兆" if cho else "") + (f"{oku:,}億" if oku else "")
        add(head + (f"{mil}百万" if mil else ""))
        if mil >= 10:
            add(head + f"{mil // 10}千" + (f"{mil % 10}百万" if mil % 10 else "万"))
    if av >= 100_000_000:  # 丸めた億と兆（「2,459億円」「1.2兆円」）は項目名が要る
        weak.add(f"{round(av / 1e8):,}億")
    if av >= 1e12:
        weak.add(f"{av / 1e12:.1f}兆")
    return strong, weak


WELL = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")  # 3 桁ごとにカンマを打った、正しい形の数値


def _run(text, i, j):
    """text[i:j] を含む、数字とカンマ（と小数点）の連なりの範囲。"""
    p, q = i, j
    while p > 0 and (text[p - 1].isdigit() or text[p - 1] == ","):
        p -= 1
    while q < len(text) and (text[q].isdigit() or (text[q] in ",." and q + 1 < len(text) and text[q + 1].isdigit())):
        q += 1
    return p, q


def _hits(text, form):
    """form が一つの数値として現れる場所。「1,245,896」や「13,018」の途中の表記は数えない。

    EDINET-Bench の表は列の区切りが消えていて、前期と当期が「396,925245,896」や「3843,018」とつながる。
    数字の連なりが一つの正しい形の数値でなければ列がつながったものとみなし、
    両側が正しい形の数値に切れる位置だけを切れ目として認める。
    """
    i = text.find(form)
    while i >= 0:
        j = i + len(form)
        if not form[-1].isdigit() and j < len(text) and text[j].isdigit():
            i = text.find(form, i + 1)  # 「75億」の後に「9千万」と続くなら、75億ちょうどではない
            continue
        p, q = _run(text, i, j)
        left, right, whole = text[p:i], text[j:q], text[p:q]
        if not left and not right:
            yield i
        elif not WELL.fullmatch(whole) and not re.fullmatch(r"(?:19|20)\d\d", whole) \
                and (not left or WELL.fullmatch(left)) and (not right or WELL.fullmatch(right)):
            yield i
        i = text.find(form, i + 1)


def _squash(s):
    return re.sub(r"[\s・･]", "", _norm(s))


def print_position(text, value_yen, labels):
    """正解の数値が印字された位置（NFKC で正規化した本文での位置）。無ければ -1。

    数値の照合は空白を残した本文で行う（空白を消すと「※1 12,109,488」が「※112,109,488」とつながる）。
    項目名の照合だけは、空白と「・」を除いて行う（「キャッシュ・フロー」と「キャッシュフロー」をそろえる）。
    """
    t = _norm(text)
    strong, weak = forms(value_yen)
    for f in strong:
        for i in _hits(t, f):
            return i
    keys = [_squash(x) for x in labels if x]
    for f in weak:
        for i in _hits(t, f):
            window = _squash(t[max(0, i - LABEL_WINDOW):i])
            if any(k and k in window for k in keys):
                return i
    return -1


def chunk_prints(text, value_yen, labels):
    """チャンクに正解の数値が印字されているか。labels は項目名（勘定科目名・行の見出し）。"""
    return print_position(text, value_yen, labels) >= 0


def first_reach(chunks, owners, company, value_yen, labels):
    """検索結果（上から順の本文と、その書類の企業）で、最初に正解の箇所が現れた順位（1 始まり）。無ければ None。"""
    for rank, (text, owner) in enumerate(zip(chunks, owners), 1):
        if owner == company and chunk_prints(text, value_yen, labels):
            return rank
    return None


# ---------------------------------------------------------------- 検査
def selftest():
    G = 245_896_000_000  # 日産自動車 2015年3月期 財務活動によるキャッシュ・フロー 245,896百万円
    N = -4_767_000_000   # シロキ工業 2015年3月期 投資活動によるキャッシュ・フロー △4,767百万円
    cases = [
        ("245,896百万円", G, None, "correct"), ("245,896 百万円", G, None, "correct"),
        ("２４５，８９６百万円", G, None, "correct"), ("245,896,000千円", G, None, "correct"),
        ("245896000000円", G, None, "correct"), ("245,896", G, None, "correct"),
        ("2,459億円", G, None, "correct"), ("2,458億円", G, None, "wrong"),
        ("2,458億96百万円", G, None, "correct"), ("2458.96億円", G, None, "correct"),
        ("2,458億9,600万円", G, None, "correct"), ("約2,459億円", G, None, "correct"),
        ("2015年3月期: 245,896百万円", G, None, "correct"), ("E02142 245,896百万円", G, None, "correct"),
        ("245,896百万円（前期 200,000百万円）", G, None, "correct"),
        ("245,896 million yen", G, None, "correct"), ("1兆2,345億円", 1_234_490_000_000, None, "correct"),
        (245896, G, "百万円", "correct"), (245896, G, None, "correct"), (245896, G, "千円", "wrong"),
        ("△4,767百万円", N, None, "correct"), ("▲4,767百万円", N, None, "correct"),
        ("マイナス4,767百万円", N, None, "correct"), ("-4,767百万円", N, None, "correct"),
        ("−4,767百万円", N, None, "correct"), ("4,767百万円", N, None, "wrong"),
        ("見つかりませんでした", G, None, "abstained"), ("記載なし", G, None, "abstained"),
        ("不明", G, None, "abstained"), ("INSUFFICIENT", G, None, "abstained"), ("N/A", G, None, "abstained"),
        ("2015年3月期の記載は見つかりませんでした", G, None, "abstained"),
        ("E02142 の財務活動によるキャッシュ・フローは記載されていません", G, None, "abstained"),
        ("渡された資料には2015年3月期の連結キャッシュ・フロー計算書が含まれていません", G, None, "abstained"),
        ("記載はないが前期は200,000百万円", G, None, "wrong"),
        ("0円", 0, None, "correct"),
    ]
    bad = 0
    for v, g, u, want in cases:
        got = read(v, g, u)
        bad += got != want
        if got != want:
            print(f"  NG read({v!r}, {g}, {u!r}) = {got}（期待 {want}）")
    L = ["投資活動によるキャッシュ・フロー"]
    reach_cases = [
        ("営業活動…投資活動によるキャッシュ・フロー△5,240△4,767財務活動によるキャッシュ・フロー", N, L, True),
        ("投資活動によるキャッシュ・フロー△520△413", -413_000_000, L, True),
        ("有形固定資産の取得による支出△520△413", -413_000_000, L, False),
        ("財務活動によるキャッシュ・フローは2,458億96百万円の収入となりました", G, ["財務活動によるキャッシュ・フロー"], True),
        ("財務活動によるキャッシュ・フローは2,459億円の収入となりました", G, ["財務活動によるキャッシュ・フロー"], True),
        ("売上高は2,459億円となりました", G, ["財務活動によるキャッシュ・フロー"], False),
        ("合計1,245,896", G, L, False),
        ("財務活動によるキャッシュ・フロー396,925245,896現金及び", G, L, True),         # 当期の列
        ("財務活動によるキャッシュ・フロー245,896301,002現金及び", G, L, True),         # 翌年の書類の前期の列
        ("財務活動によるキャッシュ・フロー1,245,8963,001現金及び", G, L, False),
        ("営業活動によるキャッシュ・フロー3843,018    (単位：百万円)", 3_018_000_000, L, True),  # 前期が 3 桁
        ("営業活動によるキャッシュ・フロー13,018", 3_018_000_000, L, False),       # 13,018 の途中
        ("営業利益589602営業外収益", 602_000_000, ["営業利益"], True),
        ("売上高※１ 10,377,898※1 12,109,488売上原価", 12_109_488_000, ["売上高"], True),
        ("営業活動によるキャッシュ・フローは、75億4千5百万円の収入", 7_545_000_000, ["営業活動によるキャッシュ・フロー"], True),
        ("営業活動によるキャッシュ・フローは、75億9千万円の収入", 7_545_000_000, ["営業活動によるキャッシュ・フロー"], False),
        ("財務活動によるキャッシュ　フロー 245,896", G, L, True),
    ]
    for text, v, labels, want in reach_cases:
        got = chunk_prints(text, v, labels)
        bad += got != want
        if got != want:
            print(f"  NG chunk_prints({text[:30]!r}…, {v}) = {got}（期待 {want}）")
    n = len(cases) + len(reach_cases)
    print(f"自己検査: {n - bad}/{n} OK")
    return bad == 0


def load_data(data_dir):
    """get_ja_data.py の出力から、書類の企業（doc_id → EDINET コード）と JA-LOOKUP の設問を読む。"""
    import json, os
    company, qs = {}, []
    with open(os.path.join(data_dir, "docs.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            company[d["doc_id"]] = d["edinet_code"]
    with open(os.path.join(data_dir, "questions_ja.jsonl"), encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x["family"] == "JA-LOOKUP":
                qs.append(x)
    return company, qs


def target(x):
    """設問の正解を (企業の EDINET コード, 円の値, 項目名) で返す。"""
    ev = x["gold"]["evidence"][0]
    return ev["edinet_code"], ev["value_yen"], [ev.get("account"), ev.get("row_label")]


def check(data_dir):
    """正解の位置を含むチャンクがすべて Reach に数えられるかと、Read が正しい書き方と間違った書き方を分けるかを確かめる。

    get_ja_data.py を --questions と --chunks を付けて走らせた出力を使う。
    """
    import collections, json, os, random
    company, qs = load_data(data_dir)
    chunks = collections.defaultdict(list)
    with open(os.path.join(data_dir, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[company[c["doc_id"]]].append(c)
    texts = {}

    def text(doc_id):
        if doc_id not in texts:
            path = os.path.join(data_dir, "text", f"{doc_id}.txt")
            texts[doc_id] = open(path, encoding="utf-8", newline="").read()
        return texts[doc_id]

    stats = collections.Counter()
    examples = []
    for x in qs:
        firm, value, labels = target(x)
        loc = x["gold_locations"][0]
        a, b = loc["span"]
        gold_hit = False
        for c in chunks[firm]:
            ch = text(c["doc_id"])[c["start"]:c["end"]]
            i = print_position(ch, value, labels)
            if c["doc_id"] == loc["doc_id"] and c["start"] <= a and b <= c["end"]:
                stats["gold_chunks"] += 1
                stats["gold_chunks_flagged"] += i >= 0
                gold_hit |= i >= 0
            elif i >= 0:
                stats["other"] += 1
                stats["other_same_doc"] += c["doc_id"] == loc["doc_id"]
                examples.append((x["id"], c["doc_id"], c["section"], _norm(ch)[max(0, i - 45):i + 20]))
        stats["questions"] += 1
        stats["questions_flagged"] += gold_hit
    print(f"Reach: 設問 {stats['questions']} 問のうち、正解の位置を含むチャンクを正解の箇所と数えた設問 "
          f"{stats['questions_flagged']} 問（チャンク {stats['gold_chunks_flagged']}/{stats['gold_chunks']}）")
    print(f"       ほかに正解の箇所と数えたチャンク {stats['other']} 個（同じ書類 {stats['other_same_doc']}・"
          f"同じ企業の別の年度の書類 {stats['other'] - stats['other_same_doc']}）。無作為に 20 個:")
    for e in random.Random(0).sample(examples, min(20, len(examples))):
        print("  ", e)

    # Read: 実際の正解の値から、正しい書き方と間違った書き方を作って分けさせる
    read_bad, read_n = [], 0
    for x in qs:
        ev = x["gold"]["evidence"][0]
        v, unit, printed = ev["value_yen"], ev["unit"], ev["printed"]
        div = {"千円": 1e3, "百万円": 1e6}[unit]
        sign = "△" if v < 0 else ""
        n = round(abs(v) / div)
        off = n + 1 if (n + 1) % 10 else n - 1  # 末尾が 0 だと「十の位で丸めた答え」と読むので避ける
        trials = [(f"{printed}{unit}", "correct"),                                   # 表示どおり
                  (f"{printed} {unit}", "correct"),
                  (f"{sign}{printed.lstrip('△-')}{'百万円' if unit == '千円' else '千円'}", "wrong"),  # 単位違い
                  (f"{sign}{off:,}{unit}", "wrong")]                                 # 最後の桁が 1 違う
        if v < 0:
            trials.append((f"{printed.lstrip('△-')}{unit}", "wrong"))               # 符号が抜けた
        if abs(v) >= 1e10 and abs(round(abs(v) / 1e8) * 1e8 - abs(v)) <= abs(v) * TOL:
            trials.append((f"{sign}{round(abs(v) / 1e8):,}億円", "correct"))          # 億円に丸めた
        for text, want in trials:
            read_n += 1
            got = read(text, v)
            if got != want:
                read_bad.append((x["id"], text, v, got, want))
    print(f"Read: 正解の値から作った {read_n} 通りの回答のうち {read_n - len(read_bad)} 通りが期待どおり")
    for e in read_bad[:10]:
        print("  NG", e)
    return stats["questions_flagged"] == stats["questions"] and not read_bad


# ---------------------------------------------------------------- 採点
def score(run_path, data_dir, out_path=None, n_boot=2000, seed=0):
    """RAG の実行記録を、六つのマスと「まず直す場所」にまとめる。

    実行記録は 1 行 1 問の JSON Lines。欄は id・chunks（回答モデルに渡した順の {"doc_id", "text"} の並び）・
    value（回答の値の欄）・unit（単位を別の欄にした場合だけ）。チャンクの区切り方は自由でよい。
    """
    import collections, json, random
    company, qs = load_data(data_dir)
    byid = {x["id"]: x for x in qs}
    rows, unknown = [], 0
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            x = byid.get(r.get("id"))
            if x is None:
                unknown += 1
                continue
            firm, value, labels = target(x)
            ch = r.get("chunks") or []
            rank = first_reach([c.get("text", "") for c in ch],
                               [c.get("edinet_code") or company.get(c.get("doc_id")) for c in ch],
                               firm, value, labels)
            rows.append({"id": x["id"], "reach_rank": rank, "read": read(r.get("value"), value, r.get("unit"))})
    if not rows:
        print("採点できる記録がありません（id が JA-LOOKUP の設問と合っているか確かめてください）。")
        return None
    if unknown:
        print(f"設問にない id の記録 {unknown} 件は除きました。")
    return summarize(rows, len(qs), out_path, n_boot, seed)


def summarize(rows, n_all, out_path=None, n_boot=2000, seed=0):
    """設問ごとの {"id", "reach_rank", "read"} から、六つのマスと「まず直す場所」を出す。

    正答できなかった割合を、検索で失敗（届かずに正答できない）と読解で失敗（届いたのに正答できない）に分け、
    その差の 95% 区間（設問を単位にしたブートストラップ）が 0 をまたがなければ、大きいほうをまず直す場所とする。
    """
    import collections, json, random
    n = len(rows)
    cells = collections.Counter((row["reach_rank"] is not None, row["read"]) for row in rows)
    print(f"設問 {n} 問（JA-LOOKUP は全 {n_all} 問）")
    if n < n_all:
        print(f"  注意: 記録のない設問が {n_all - n} 問あります。ほかと比べるときは全 {n_all} 問を回してください。")
    def pad(s, w, right=False):  # 全角を 2 桁と数えて揃える
        width = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))
        return (" " * (w - width) + str(s)) if right else (str(s) + " " * (w - width))

    print(pad("", 14) + "".join(pad(h, 8, True) for h in ("正答", "無回答", "誤答")))
    for reached, name in ((True, "届いた"), (False, "届かなかった")):
        print(pad(name, 14) + "".join(pad(cells[(reached, v)], 8, True) for v in ("correct", "abstained", "wrong")))
    retr = [row["reach_rank"] is None and row["read"] != "correct" for row in rows]
    rdg = [row["reach_rank"] is not None and row["read"] != "correct" for row in rows]
    diff = (sum(retr) - sum(rdg)) / n
    rng = random.Random(seed)
    boot = sorted(sum(retr[i] - rdg[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(n_boot))
    lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]
    where = ("直す場所なし（失敗した設問がない）" if not any(retr) and not any(rdg)
             else "検索" if lo > 0 else "読解" if hi < 0 else "決められない（区間が 0 をまたぐ）")
    print(f"正答率 {sum(row['read'] == 'correct' for row in rows) / n:.3f}")
    print(f"検索で失敗（届かずに正答できない）{sum(retr) / n:.3f}・読解で失敗（届いたのに正答できない）{sum(rdg) / n:.3f}")
    print(f"差（検索 − 読解）{diff:+.3f}、95% 区間 [{lo:+.3f}, {hi:+.3f}] → まず直す場所: {where}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"n": n, "cells": {f"{'R' if k[0] else 'N'}_{k[1]}": v for k, v in cells.items()},
            "retrieval_fail": sum(retr) / n, "reading_fail": sum(rdg) / n, "diff_ci": [lo, hi], "fix": where}


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Reach-Read Method の日本語版（照合と採点の規則）")
    ap.add_argument("--selftest", action="store_true", help="手で作った例で規則を確かめる")
    ap.add_argument("--check", metavar="DATA_DIR", help="get_ja_data.py の出力で規則を確かめる")
    ap.add_argument("--score", metavar="RUN_JSONL", help="RAG の実行記録を採点する")
    ap.add_argument("--data", metavar="DATA_DIR", default="ja_data", help="get_ja_data.py の出力（既定 ja_data）")
    ap.add_argument("--out", metavar="JSONL", help="設問ごとの Reach と Read を書き出す")
    args = ap.parse_args()
    ok = True
    if args.selftest:
        ok &= selftest()
    if args.check:
        ok &= check(args.check)
    if args.score:
        ok &= score(args.score, args.data, args.out) is not None
    if not (args.selftest or args.check or args.score):
        ap.print_help()
    sys.exit(0 if ok else 1)
