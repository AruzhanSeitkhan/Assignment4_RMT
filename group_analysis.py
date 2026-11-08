#!/usr/bin/env python3
"""
Assignment 3 - CPU-only evidence for the Kazakh QA comparison (no LLM, no training, no invented data).

Uses the SAME eligibility rules as qa_compare.py (imports load_group / normalise / score) and produces:
  O2  comparability of the two eligible groups (lengths, answer types, script, question-passage overlap)
  O2  frozen main sample (100 matched pairs, pilot questions excluded) + covariate balance
  O3  reference condition: a deterministic lexical sliding-window QA baseline scored with the same EM/F1
  O4  automatic data-issue screening (question/answer type mismatch, answer repeated in question,
      non-interrogative "questions") with rates and examples for manual review

Usage: python3 group_analysis.py --data <KazQAD repo> --pilot results_pilot/selected.json --out results_a3
"""
import argparse, collections, json, math, random, re, platform, time
from pathlib import Path
import numpy as np
from scipy import stats
import qa_compare as q

SEED = 42
KZ = set("әғқңөұүһі")
INTERROG = re.compile(r"\b(қай|қайда|қайдан|қайсы|қайсысы|қашан|кім|кімнің|кімге|кімді|не|нені|неге|немен|неше|нешінші|қанша|қандай|қалай|неліктен|нешеде)\w*", re.I)
NUM_Q = re.compile(r"\b(қай жылы|қашан|неше|нешінші|қанша|қай ғасыр|қай күні)", re.I)
NUM_WORDS = re.compile(r"\b(бір|екі|үш|төрт|бес|алты|жеті|сегіз|тоғыз|он|жүз|мың|ғасыр|жыл)\w*", re.I)

def tok(s): return q.normalise(s).split()

def features(r):
    qt, ct = tok(r["question"]), tok(r["context"])
    ref = r["references"][0]
    letters = [c for c in (r["question"] + " " + r["context"]).lower() if c.isalpha()]
    cyr = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
    lat = sum(1 for c in letters if c.isascii())
    kz = sum(1 for c in letters if c in KZ)
    cset = set(ct)
    content_q = [w for w in qt if len(w) > 2]
    overlap = sum(1 for w in content_q if w in cset) / max(len(content_q), 1)
    pos = q.normalise(r["context"]).find(q.normalise(ref))
    return dict(q_words=len(qt), c_words=len(ct), a_words=len(tok(ref)), n_refs=len(r["references"]),
                latin_pct=100 * lat / max(len(letters), 1), kz_per100=100 * kz / max(cyr, 1),
                q_passage_overlap=overlap, answer_rel_pos=pos / max(len(q.normalise(r["context"])), 1),
                has_qmark=r["question"].strip().endswith("?"), has_interrog=bool(INTERROG.search(q.normalise(r["question"]))))

def issues(r):
    qn = q.normalise(r["question"]); refs = [q.normalise(x) for x in r["references"]]
    num_q = bool(NUM_Q.search(qn))
    num_ans = any(any(ch.isdigit() for ch in x) or NUM_WORDS.search(x) for x in refs)
    return dict(numeric_question_non_numeric_answer=num_q and not num_ans,
                answer_contained_in_question=any(len(x) > 2 and re.search(r"\b" + re.escape(x) + r"\b", qn) for x in refs),
                no_interrogative_form=(not r["question"].strip().endswith("?")) and not INTERROG.search(qn))

# ---------------- lexical sliding-window baseline (Richardson et al., 2013 idea, span version)
def baseline_answer(r, max_len=4, k=6):
    ctx_raw = r["context"].split()
    ctx = [q.normalise(w) for w in ctx_raw]
    qset = set(w for w in tok(r["question"]) if len(w) > 2)
    df = collections.Counter(ctx)
    idf = {w: math.log(1 + len(ctx) / df[w]) for w in df}
    best, best_s = None, -1.0
    for i in range(len(ctx)):
        for L in range(1, max_len + 1):
            j = i + L
            if j > len(ctx): break
            span = ctx[i:j]
            if not "".join(span) or any(w in qset for w in span): continue  # answers rarely repeat question words
            win = ctx[max(0, i - k):i] + ctx[j:j + k]
            s = sum(idf.get(w, 0) for w in set(win) if w in qset) - 0.05 * L
            if s > best_s: best, best_s = (i, j), s
    return " ".join(ctx_raw[best[0]:best[1]]) if best else ""

def boot_mean(v, B=2000, seed=SEED):
    rng = random.Random(seed); n = len(v)
    m = sorted(sum(rng.choices(v, k=n)) / n for _ in range(B))
    return [m[int(.025 * B)], m[int(.975 * B) - 1]]

def boot_gap(a, b, B=2000, seed=SEED):
    rng = random.Random(seed)
    g = sorted(sum(rng.choices(a, k=len(a))) / len(a) - sum(rng.choices(b, k=len(b))) / len(b) for _ in range(B))
    return [g[int(.025 * B)], g[int(.975 * B) - 1]]

def cliffs_delta(a, b):
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1

def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s else 0.0

def wilson(k, n, z=1.96):
    if n == 0: return [0, 0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default="results_a3"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True); t0 = time.time()
    root = Path(a.data)
    paths = {"original": root / "data/reading-comprehension/kazqad-reading-comprehension-v1.0-kk-test.jsonl",
             "translated": root / "data/supplementary/nq-translate-kk/nq-reading-comprehension-translate-kk.jsonl.gz"}
    G = {g: q.load_group(p, g)[0] for g, p in paths.items()}
    shared = {q.normalise(x["question"]) for x in G["original"]} & {q.normalise(x["question"]) for x in G["translated"]}
    pilot = {q.normalise(x["question"]) for x in json.loads(Path(a.pilot).read_text(encoding="utf-8"))}
    for g in G: G[g] = [x for x in G[g] if q.normalise(x["question"]) not in shared | pilot]
    res = {"python": platform.python_version(), "numpy": np.__version__, "scipy": stats.__name__ and __import__("scipy").__version__,
           "seed": SEED, "n_eligible_after_pilot_exclusion": {g: len(v) for g, v in G.items()}}

    # ---- O2: comparability of all eligible items
    F = {g: [features(x) for x in v] for g, v in G.items()}
    comp = {}
    for key in ["q_words", "c_words", "a_words", "n_refs", "latin_pct", "kz_per100", "q_passage_overlap", "answer_rel_pos"]:
        A_, B_ = [f[key] for f in F["original"]], [f[key] for f in F["translated"]]
        p = stats.mannwhitneyu(A_, B_).pvalue
        comp[key] = dict(original_median=float(np.median(A_)), original_mean=float(np.mean(A_)),
                         translated_median=float(np.median(B_)), translated_mean=float(np.mean(B_)),
                         mannwhitney_p=float(p), cliffs_delta=float(cliffs_delta(A_, B_)))
    for key in ["has_qmark", "has_interrog"]:
        k1 = sum(f[key] for f in F["original"]); k2 = sum(f[key] for f in F["translated"])
        comp[key] = dict(original_pct=100 * k1 / len(F["original"]), translated_pct=100 * k2 / len(F["translated"]))
    at = {g: collections.Counter(x["answer_type"] for x in v) for g, v in G.items()}
    table = [[at["original"][t], at["translated"][t]] for t in ["number", "short", "long"]]
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n_all = sum(map(sum, table))
    comp["answer_type"] = dict(counts={g: dict(at[g]) for g in at}, chi2=float(chi2), p=float(p),
                               cramers_v=float(math.sqrt(chi2 / (n_all * (min(len(table), 2) - 1)))))
    res["O2_comparability_all_eligible"] = comp

    # ---- O2: frozen main sample (same stratified matching as qa_compare.prepare)
    by = {g: collections.defaultdict(list) for g in G}
    for g in G:
        for x in G[g]: by[g][x["stratum"]].append(x)
    rng = random.Random(SEED); pairs = []
    for key in sorted(set(by["original"]) & set(by["translated"])):
        o, t = by["original"][key][:], by["translated"][key][:]
        rng.shuffle(o); rng.shuffle(t); pairs.extend(zip(o, t))
    rng.shuffle(pairs)
    main_rows = [r for pr in pairs[:a.n] for r in pr]
    q.dump(out / "selected_main.json", main_rows)
    res["main_sample"] = dict(available_matched_pairs=len(pairs), n_per_group=a.n, selected_sha256=q.sha(out / "selected_main.json"),
                              strata=dict(collections.Counter(r["stratum"] for r in main_rows if r["group"] == "original")))
    bal = {}
    for key in ["q_words", "c_words", "a_words", "q_passage_overlap", "latin_pct", "kz_per100"]:
        o = [features(r)[key] for r in main_rows if r["group"] == "original"]
        t = [features(r)[key] for r in main_rows if r["group"] == "translated"]
        oa = [f[key] for f in F["original"]]; ta = [f[key] for f in F["translated"]]
        bal[key] = dict(smd_all_eligible=smd(oa, ta), smd_main_sample=smd(o, t))
    res["main_sample_balance_SMD"] = bal

    # ---- O3 reference condition: lexical baseline, same scoring as the LLM
    pred_rows = []
    def run_baseline(rows, tag):
        out_ = {}
        for g in ["original", "translated"]:
            sub = [r for r in rows if r["group"] == g]
            em, f1 = [], []
            for r in sub:
                p_ = baseline_answer(r); e, f = q.score(p_, r["references"]); em.append(e); f1.append(f)
                if tag == "main": pred_rows.append(dict(group=g, id=r["id"], question=r["question"], prediction=p_,
                                                        references=r["references"], exact_match=e, token_f1=round(f, 4)))
            out_[g] = dict(n=len(sub), em=sum(em) / len(em), em_ci95=boot_mean(em), f1=sum(f1) / len(f1), f1_ci95=boot_mean(f1),
                           _em=em, _f1=f1)
        o, t = out_["original"], out_["translated"]
        k1, k2 = int(sum(o["_em"])), int(sum(t["_em"]))
        fisher_p = stats.fisher_exact([[k1, o["n"] - k1], [k2, t["n"] - k2]]).pvalue
        gaps = dict(em_gap=o["em"] - t["em"], em_gap_ci95=boot_gap(o["_em"], t["_em"]), fisher_exact_p_em=float(fisher_p),
                    f1_gap=o["f1"] - t["f1"], f1_gap_ci95=boot_gap(o["_f1"], t["_f1"]),
                    mannwhitney_p_f1=float(stats.mannwhitneyu(o["_f1"], t["_f1"]).pvalue))
        for g in out_: out_[g].pop("_em"); out_[g].pop("_f1")
        return dict(groups=out_, gaps=gaps)
    res["O3_lexical_baseline_main_sample"] = run_baseline(main_rows, "main")
    rng2 = random.Random(SEED)
    big = rng2.sample(G["original"], 1000) + rng2.sample(G["translated"], 1000)
    res["O3_lexical_baseline_random_1000_per_group_unmatched"] = run_baseline(big, "big")
    q.write_csv(out / "baseline_predictions_main.csv",
                [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in r.items()} for r in pred_rows])

    # ---- O4 automatic issue screening (all eligible) + examples
    iss = {}
    ex = collections.defaultdict(list)
    for g, v in G.items():
        cnt = collections.Counter()
        for r in v:
            for k_, flag in issues(r).items():
                if flag:
                    cnt[k_] += 1
                    if len(ex[f"{g}:{k_}"]) < 4: ex[f"{g}:{k_}"].append((r["question"], r["references"][:2]))
        iss[g] = {k_: dict(count=cnt[k_], pct=100 * cnt[k_] / len(v), ci95=[100 * x for x in wilson(cnt[k_], len(v))])
                  for k_ in ["numeric_question_non_numeric_answer", "answer_contained_in_question", "no_interrogative_form"]}
    res["O4_issue_screening_all_eligible"] = iss
    res["O4_issue_examples"] = ex
    # review sheet for the main sample
    review = []
    for r in main_rows:
        f = issues(r)
        review.append(dict(group=r["group"], id=r["id"], stratum=r["stratum"], question=r["question"],
                           references=json.dumps(r["references"], ensure_ascii=False), title=r["title"],
                           auto_flags=";".join(k for k, v in f.items() if v), review_topic="", review_answer_supported="",
                           review_ambiguity="", review_translation_issue=""))
    q.write_csv(out / "main_sample_review.csv", review)
    res["runtime_s"] = round(time.time() - t0, 1)
    q.dump(out / "a3_results.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "O4_issue_examples"}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
