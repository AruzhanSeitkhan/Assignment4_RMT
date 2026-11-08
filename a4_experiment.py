#!/usr/bin/env python3
"""
Assignment 4 - final experiment runner and analysis for the Kazakh QA comparison.

Conditions (frozen in configs/protocol.json):
  lexical      simple baseline: IDF-weighted sliding-window span selection (CPU, deterministic)
  xlmr         strong literature baseline: deepset/xlm-roberta-large-squad2 extractive reader
  qwen_open    primary condition: Qwen2.5-1.5B-Instruct with passage (run by qa_compare.py run)
  qwen_closed  ablation: the same model WITHOUT the passage

Commands
  python a4_experiment.py lexical
  python a4_experiment.py xlmr          (GPU recommended)
  python qa_compare.py run --out results_main --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
  python a4_experiment.py import-qwen
  python a4_experiment.py qwen-closed   (GPU recommended)
  python a4_experiment.py analyse       -> results_main/analysis/*  (tables, statistics, figures, error sheet)
  python a4_experiment.py kappa         -> after both coders fill error_analysis_sheet.csv

Nothing here tunes anything on the test items. All settings come from configs/protocol.json.
"""
import argparse, collections, csv, hashlib, json, math, platform, random, re, sys, time
from pathlib import Path

import qa_compare as q

ROOT = Path(__file__).resolve().parent
PROTO = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
MAIN = ROOT / "results_main"
AN = MAIN / "analysis"
SEED = PROTO["seed"]
CONDS = ["lexical", "xlmr", "qwen_open", "qwen_closed"]
LABEL = {"lexical": "Lexical baseline", "xlmr": "XLM-R-large SQuAD2", "qwen_open": "Qwen2.5-1.5B (passage)", "qwen_closed": "Qwen2.5-1.5B (no passage)"}


# ------------------------------------------------------------------ helpers
def load_selected():
    p = MAIN / "selected.json"
    h = q.sha(p)
    if h != PROTO["data"]["test_sha256"]:
        raise SystemExit(f"Frozen test file hash mismatch: {h}")
    return json.loads(p.read_text(encoding="utf-8"))

def containment(pred, refs):
    p = " " + q.normalise(pred) + " "
    return float(any(q.normalise(r) and (" " + q.normalise(r) + " ") in p for r in refs))

def finish(rows, name, meta):
    out = MAIN / f"pred_{name}.json"
    if out.exists():
        raise SystemExit(f"{out} exists - archive it before re-running (protocol: never overwrite).")
    for r in rows:
        r["exact_match"], r["token_f1"] = q.score(r["prediction"], r["references"])
        r["containment"] = containment(r["prediction"], r["references"])
    q.dump(out, rows)
    meta.update(condition=name, n=len(rows), python=platform.python_version(), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                test_sha256=PROTO["data"]["test_sha256"])
    q.dump(MAIN / f"manifest_{name}.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

def lib_versions(names):
    import importlib.metadata as m
    return {n: m.version(n) for n in names}


# ------------------------------------------------------------------ conditions
def run_lexical(_):
    import group_analysis as ga
    rows, t0 = [], time.time()
    for r in load_selected():
        s = time.perf_counter(); p = ga.baseline_answer(r)
        rows.append(r | dict(prediction=p, seconds=time.perf_counter() - s))
    finish(rows, "lexical", dict(method="sliding-window lexical span, max_len=4, k=6", device="cpu", runtime_seconds=time.time() - t0))

def run_xlmr(_):
    import torch
    from transformers import pipeline
    from huggingface_hub import model_info
    cfg = PROTO["conditions"]["xlmr"]
    rev = model_info(cfg["model"]).sha
    dev = 0 if torch.cuda.is_available() else -1
    qa = pipeline("question-answering", model=cfg["model"], revision=rev, device=dev)
    rows, t0 = [], time.time()
    for r in load_selected():
        s = time.perf_counter()
        a = qa(question=r["question"], context=r["context"], max_answer_len=cfg["max_answer_len"],
               handle_impossible_answer=cfg["handle_impossible_answer"])
        rows.append(r | dict(prediction=a["answer"].strip(), reader_score=float(a["score"]), seconds=time.perf_counter() - s))
        print(r["group"], r["id"], repr(a["answer"]), flush=True)
    finish(rows, "xlmr", dict(model=cfg["model"], model_revision=rev, device="cuda" if dev == 0 else "cpu",
                              versions=lib_versions(["torch", "transformers"]), runtime_seconds=time.time() - t0))

def run_qwen_closed(_):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    cfg = PROTO["conditions"]["qwen_closed"]
    set_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], torch_dtype=dtype).to(dev).eval()
    rows, t0 = [], time.time()
    for r in load_selected():
        msgs = [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": f"Сұрақ: {r['question']}\nЖауап:"}]
        inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dev)
        s = time.perf_counter()
        with torch.inference_mode():
            o = model.generate(**inp, do_sample=False, max_new_tokens=cfg["max_new_tokens"], pad_token_id=tok.eos_token_id)
        new = o[0, inp["input_ids"].shape[-1]:]
        rows.append(r | dict(prediction=tok.decode(new, skip_special_tokens=True).strip(), generated_tokens=len(new),
                             reached_generation_limit=len(new) >= cfg["max_new_tokens"], seconds=time.perf_counter() - s))
        print(r["group"], r["id"], repr(rows[-1]["prediction"][:60]), flush=True)
    finish(rows, "qwen_closed", dict(model=cfg["model"], model_revision=cfg["revision"], device=dev, dtype=str(dtype),
                                     versions=lib_versions(["torch", "transformers"]), runtime_seconds=time.time() - t0))

def import_qwen(_):
    preds = json.loads((MAIN / "predictions.json").read_text(encoding="utf-8"))
    man = json.loads((MAIN / "run_manifest.json").read_text(encoding="utf-8"))
    sel = {(r["group"], r["id"]) for r in load_selected()}
    if {(r["group"], r["id"]) for r in preds} != sel:
        raise SystemExit("predictions.json does not cover exactly the frozen test items")
    per = man["runtime_seconds"] / len(preds)
    rows = [r | dict(seconds=per) for r in preds]
    finish(rows, "qwen_open", dict(source="qa_compare.py run", model=man["model"], model_revision=man["model_revision"],
                                   device=man["device"], versions=man["versions"], runtime_seconds=man["runtime_seconds"],
                                   note="per-item seconds = total runtime / n (qa_compare logs total only)"))


# ------------------------------------------------------------------ statistics
def mean(v): return sum(v) / len(v) if v else float("nan")

def boot_ci(v, B=2000, seed=SEED):
    r = random.Random(seed); n = len(v)
    m = sorted(mean(r.choices(v, k=n)) for _ in range(B))
    return [m[int(.025 * B)], m[int(.975 * B) - 1]]

def boot_gap(a, b, B=2000, seed=SEED):
    r = random.Random(seed)
    g = sorted(mean(r.choices(a, k=len(a))) - mean(r.choices(b, k=len(b))) for _ in range(B))
    return [g[int(.025 * B)], g[int(.975 * B) - 1]]

def boot_did(xo, xt, yo, yt, B=2000, seed=SEED):
    """(x_orig - x_trans) - (y_orig - y_trans); x and y are scores of two conditions on the SAME items."""
    r = random.Random(seed); no, nt = len(xo), len(xt); d = []
    for _ in range(B):
        io = [r.randrange(no) for _ in range(no)]; it = [r.randrange(nt) for _ in range(nt)]
        d.append((mean([xo[i] for i in io]) - mean([xt[i] for i in it])) - (mean([yo[i] for i in io]) - mean([yt[i] for i in it])))
    d.sort(); return [d[int(.025 * B)], d[int(.975 * B) - 1]]

def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i]); adj = [0.0] * len(pvals); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - k) * pvals[i])); adj[i] = run
    return adj

def cohens_h(p1, p2): return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

def kappa(a, b):
    labs = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def auto_code(r):
    p = q.normalise(r["prediction"])
    if not p: return "empty_output"
    if r.get("reached_generation_limit"): return "hit_length_limit"
    if r["containment"] == 1: return "correct_with_extra_wording"
    if r["token_f1"] > 0: return "partial_overlap"
    return "wrong_or_unsupported"


def analyse(_):
    from scipy import stats
    import group_analysis as ga
    AN.mkdir(exist_ok=True)
    log = open(AN / "analysis_log.txt", "w", encoding="utf-8")
    def L(*x):
        s = " ".join(str(y) for y in x); print(s); log.write(s + "\n")
    L("analysis", time.strftime("%Y-%m-%d %H:%M:%S"), "python", platform.python_version(), "scipy", __import__("scipy").__version__, "seed", SEED)
    sel = load_selected()
    P = {}
    for c in CONDS:
        f = MAIN / f"pred_{c}.json"
        if f.exists():
            rows = json.loads(f.read_text(encoding="utf-8"))
            P[c] = {(r["group"], r["id"]): r for r in rows}
            L(f"loaded {c}: {len(rows)} items, sha256 {q.sha(f)[:16]}")
        else:
            L(f"MISSING condition {c} - reported as not run")
    keys = {g: [(r["group"], r["id"]) for r in sel if r["group"] == g] for g in ["original", "translated"]}
    res = {"conditions_run": list(P), "n": {g: len(v) for g, v in keys.items()}}

    # ---- main table
    main = []
    for c in P:
        for g in ["original", "translated"]:
            row = dict(condition=c, label=LABEL[c], group=g, n=len(keys[g]))
            for m in ["exact_match", "token_f1", "containment"]:
                v = [P[c][k][m] for k in keys[g]]
                row[m] = mean(v); row[m + "_ci95"] = boot_ci(v)
            row["seconds_median"] = sorted(P[c][k].get("seconds", float("nan")) for k in keys[g])[len(keys[g]) // 2]
            main.append(row)
    res["main_table"] = main
    with open(AN / "main_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["condition", "group", "n", "EM", "EM_CI95", "F1", "F1_CI95", "containment", "containment_CI95", "median_s_per_item"])
        for r in main:
            w.writerow([r["label"], r["group"], r["n"], f"{r['exact_match']:.3f}", "[%.3f, %.3f]" % tuple(r["exact_match_ci95"]),
                        f"{r['token_f1']:.3f}", "[%.3f, %.3f]" % tuple(r["token_f1_ci95"]), f"{r['containment']:.3f}",
                        "[%.3f, %.3f]" % tuple(r["containment_ci95"]), f"{r['seconds_median']:.3f}"])

    # ---- origin gap per condition (independent groups) + Holm over secondary gap tests
    gaps = {}
    for c in P:
        o = {m: [P[c][k][m] for k in keys["original"]] for m in ["exact_match", "token_f1", "containment"]}
        t = {m: [P[c][k][m] for k in keys["translated"]] for m in ["exact_match", "token_f1", "containment"]}
        ko, kt = int(sum(o["exact_match"])), int(sum(t["exact_match"]))
        fisher = stats.fisher_exact([[ko, len(o["exact_match"]) - ko], [kt, len(t["exact_match"]) - kt]])
        mw = stats.mannwhitneyu(o["token_f1"], t["token_f1"], alternative="two-sided")
        gaps[c] = dict(
            em_orig=mean(o["exact_match"]), em_trans=mean(t["exact_match"]), em_gap=mean(o["exact_match"]) - mean(t["exact_match"]),
            em_gap_ci95=boot_gap(o["exact_match"], t["exact_match"]), fisher_odds_ratio=float(fisher.statistic), fisher_p=float(fisher.pvalue),
            cohens_h=cohens_h(mean(o["exact_match"]), mean(t["exact_match"])),
            f1_gap=mean(o["token_f1"]) - mean(t["token_f1"]), f1_gap_ci95=boot_gap(o["token_f1"], t["token_f1"]),
            mannwhitney_U=float(mw.statistic), mannwhitney_p=float(mw.pvalue),
            rank_biserial_r=2 * float(mw.statistic) / (len(o["token_f1"]) * len(t["token_f1"])) - 1,
            containment_gap=mean(o["containment"]) - mean(t["containment"]), containment_gap_ci95=boot_gap(o["containment"], t["containment"]))
    sec = [c for c in gaps if c != "qwen_open"] + (["qwen_open"] if "qwen_open" in gaps else [])
    adj = holm([gaps[c]["mannwhitney_p"] for c in sec])
    for c, a in zip(sec, adj): gaps[c]["mannwhitney_p_holm"] = a
    res["origin_gaps"] = gaps

    # ---- hypotheses
    H = {}
    if "qwen_open" in gaps:
        g = gaps["qwen_open"]
        H["H1"] = dict(test="Fisher exact, two-sided, EM", p=g["fisher_p"], odds_ratio=g["fisher_odds_ratio"], em_gap=g["em_gap"],
                       ci95=g["em_gap_ci95"], cohens_h=g["cohens_h"],
                       verdict="supported" if g["fisher_p"] < PROTO["alpha"] else "not supported (no reliable EM difference detected)")
    if "qwen_open" in P and "lexical" in P:
        f = lambda c, gr: [P[c][k]["token_f1"] for k in keys[gr]]
        did = mean(f("qwen_open", "original")) - mean(f("qwen_open", "translated")) - (mean(f("lexical", "original")) - mean(f("lexical", "translated")))
        ci = boot_did(f("qwen_open", "original"), f("qwen_open", "translated"), f("lexical", "original"), f("lexical", "translated"))
        H["H2"] = dict(test="bootstrap CI of difference-in-differences (token F1)", did=did, ci95=ci,
                       verdict="supported" if (ci[0] > 0 or ci[1] < 0) else "not supported (Qwen gap not distinguishable from lexical-baseline gap)")
    if "qwen_closed" in gaps:
        g = gaps["qwen_closed"]
        H["H3"] = dict(test="bootstrap CI of closed-book F1 gap (original - translated)", f1_gap=g["f1_gap"], ci95=g["f1_gap_ci95"],
                       mannwhitney_p=g["mannwhitney_p"],
                       verdict="supported" if g["f1_gap_ci95"][1] < 0 else ("opposite direction" if g["f1_gap_ci95"][0] > 0 else "not supported"))
    res["hypotheses"] = H

    # ---- McNemar between methods on the same items, Holm over all pairs
    mc = []
    pairs = [(a, b) for a, b in [("qwen_open", "xlmr"), ("qwen_open", "lexical"), ("xlmr", "lexical")] if a in P and b in P]
    for g in ["original", "translated"]:
        for a, b in pairs:
            ea = [P[a][k]["exact_match"] for k in keys[g]]; eb = [P[b][k]["exact_match"] for k in keys[g]]
            n01 = sum(1 for x, y in zip(ea, eb) if x == 0 and y == 1); n10 = sum(1 for x, y in zip(ea, eb) if x == 1 and y == 0)
            p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            mc.append(dict(group=g, a=a, b=b, a_only_correct=n10, b_only_correct=n01, em_diff=mean(ea) - mean(eb), mcnemar_exact_p=p))
    for m, a in zip(mc, holm([m["mcnemar_exact_p"] for m in mc])): m["p_holm"] = a
    res["mcnemar"] = mc

    # ---- subgroups
    sub = []
    for c in P:
        for g in ["original", "translated"]:
            for dim, fn in [("answer_type", lambda r: r["answer_type"]), ("passage_bin", lambda r: f"{50*(r['context_words']//50)}-{50*(r['context_words']//50)+49} words")]:
                buckets = collections.defaultdict(list)
                for k in keys[g]: buckets[fn(P[c][k])].append(P[c][k])
                for lev, rs in sorted(buckets.items()):
                    sub.append(dict(condition=c, group=g, dimension=dim, level=lev, n=len(rs), em=mean([r["exact_match"] for r in rs]),
                                    f1=mean([r["token_f1"] for r in rs]), containment=mean([r["containment"] for r in rs])))
    res["subgroups"] = sub

    # ---- sensitivity: excluding auto-flagged items
    flagged = {(r["group"], r["id"]) for r in sel if any(ga.issues(r).values())}
    sens = []
    for c in P:
        for g in ["original", "translated"]:
            ks = [k for k in keys[g] if k not in flagged]
            sens.append(dict(condition=c, group=g, n=len(ks), em=mean([P[c][k]["exact_match"] for k in ks]), f1=mean([P[c][k]["token_f1"] for k in ks]),
                             containment=mean([P[c][k]["containment"] for k in ks])))
    res["sensitivity_excluding_flagged"] = dict(n_flagged=len(flagged), rows=sens)

    # ---- error analysis sheet (qwen_open failures)
    if "qwen_open" in P:
        rng = random.Random(SEED); sheet = []; codes = collections.Counter()
        for g in ["original", "translated"]:
            fails = [P["qwen_open"][k] for k in keys[g] if P["qwen_open"][k]["exact_match"] == 0]
            for r in fails: codes[(g, auto_code(r))] += 1
            for r in rng.sample(fails, min(30, len(fails))):
                sheet.append(dict(group=g, id=r["id"], question=r["question"], references=json.dumps(r["references"], ensure_ascii=False),
                                  prediction=r["prediction"], token_f1=round(r["token_f1"], 3), auto_code=auto_code(r),
                                  coder1="", coder2="", final_code="", note=""))
        q.write_csv(AN / "error_analysis_sheet.csv", sheet)
        res["auto_error_codes_all_failures"] = {f"{g}|{c}": n for (g, c), n in sorted(codes.items())}
        res["error_scheme"] = ["correct_with_extra_wording", "wrong_entity_or_number", "unsupported_or_hallucinated", "reference_error",
                               "translation_problem", "ambiguous_question", "inflection_or_format_mismatch", "empty_or_limit"]

    # ---- efficiency
    res["efficiency"] = {c: json.loads((MAIN / f"manifest_{c}.json").read_text(encoding="utf-8")) for c in P if (MAIN / f"manifest_{c}.json").exists()}
    q.dump(AN / "statistics.json", res)
    L(json.dumps({k: res[k] for k in ["hypotheses", "origin_gaps"]}, ensure_ascii=False, indent=1))
    figures(res, P, keys)
    L("analysis finished")

def figures(res, P, keys):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S1, S2, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID})
    conds = [c for c in CONDS if c in P]
    # F1 and EM by condition x group
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=200, sharey=True)
    for ax, m, t in zip(axes, ["exact_match", "token_f1"], ["Exact match (primary)", "Token F1"]):
        for j, (g, col) in enumerate([("original", S1), ("translated", S2)]):
            rows = {r["condition"]: r for r in res["main_table"] if r["group"] == g}
            xs = [i + (j - .5) * .38 for i in range(len(conds))]; vs = [rows[c][m] for c in conds]
            lo = [v - rows[c][m + "_ci95"][0] for c, v in zip(conds, vs)]; hi = [rows[c][m + "_ci95"][1] - v for c, v in zip(conds, vs)]
            ax.bar(xs, vs, .36, color=col, label=g.capitalize()); ax.errorbar(xs, vs, yerr=[lo, hi], fmt="none", ecolor=INK2, capsize=2, lw=1)
            for x, v, h in zip(xs, vs, hi): ax.text(x, v + h + .01, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([LABEL[c].replace(" (", "\n(") for c in conds], fontsize=6.8); ax.set_title(t, fontsize=9)
    axes[0].set_ylabel("Score (95% bootstrap CI)"); axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(AN / "fig1_scores_by_condition.png"); plt.close(fig)
    # forest plot of F1 gaps
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    for i, c in enumerate(conds):
        g = res["origin_gaps"][c]; ax.plot(g["f1_gap_ci95"], [i, i], color=S1, lw=2); ax.scatter([g["f1_gap"]], [i], color=S1, s=36, zorder=3)
        ax.text(g["f1_gap_ci95"][1] + .01, i, f"{g['f1_gap']:+.3f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=.8); ax.set_yticks(range(len(conds))); ax.set_yticklabels([LABEL[c] for c in conds]); ax.invert_yaxis()
    ax.set_xlabel("Token-F1 gap, original − translated (95% bootstrap CI)")
    fig.tight_layout(); fig.savefig(AN / "fig2_origin_gap_forest.png"); plt.close(fig)
    # subgroup by answer type for qwen_open (or first available)
    c = "qwen_open" if "qwen_open" in P else conds[0]
    rows = [r for r in res["subgroups"] if r["condition"] == c and r["dimension"] == "answer_type"]
    levels = ["number", "short", "long"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
    for j, (g, col) in enumerate([("original", S1), ("translated", S2)]):
        d = {r["level"]: r for r in rows if r["group"] == g}
        xs = [i + (j - .5) * .38 for i in range(3)]
        ax.bar(xs, [d.get(l, {}).get("f1", 0) for l in levels], .36, color=col, label=g.capitalize())
        for x, l in zip(xs, levels): ax.text(x, d.get(l, {}).get("f1", 0) + .01, f"n={d.get(l, {}).get('n', 0)}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Number answer", "Short answer (≤3 words)", "Long answer"]); ax.set_ylabel("Mean token F1")
    ax.set_title(f"{LABEL[c]}: token F1 by answer type", fontsize=9); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(AN / "fig3_subgroup_answer_type.png"); plt.close(fig)

def run_kappa(_):
    rows = list(csv.DictReader(open(AN / "error_analysis_sheet.csv", encoding="utf-8-sig")))
    done = [r for r in rows if r["coder1"] and r["coder2"]]
    if not done: raise SystemExit("fill coder1 and coder2 first")
    k = kappa([r["coder1"] for r in done], [r["coder2"] for r in done])
    final = collections.Counter((r["group"], r["final_code"] or r["coder1"]) for r in done)
    out = dict(n_double_coded=len(done), cohens_kappa=k, counts={f"{g}|{c}": n for (g, c), n in sorted(final.items())})
    q.dump(AN / "error_analysis_summary.json", out); print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["lexical", "xlmr", "qwen-closed", "import-qwen", "analyse", "kappa"])
    a = ap.parse_args()
    {"lexical": run_lexical, "xlmr": run_xlmr, "qwen-closed": run_qwen_closed, "import-qwen": import_qwen, "analyse": analyse, "kappa": run_kappa}[a.cmd](a)
