"""Frozen-model Kazakh QA comparison. No training and no fabricated outputs."""
import argparse, collections, csv, gzip, hashlib, importlib.metadata, json
import math, platform, random, re, subprocess, time, unicodedata
from pathlib import Path

SEED = 42
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
DATA_COMMIT = 'ab8684a667109f2a7345f4f89ab3b123e8d78e1f'
SYSTEM = ('Сұраққа тек берілген мәтінге сүйеніп жауап бер. '
          'Мәтіннен қысқа жауап үзіндісін ғана жаз. Түсіндірме қоспа.')

def normalise(text):
    text = unicodedata.normalize('NFC', text).casefold()
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return ' '.join(text.split())

def score(prediction, references):
    p = normalise(prediction).split()
    em, f1 = 0.0, 0.0
    for reference in references:
        r = normalise(reference).split()
        em = max(em, float(p == r))
        overlap = sum((collections.Counter(p) & collections.Counter(r)).values())
        value = 2 * overlap / (len(p) + len(r)) if p and r else float(p == r)
        f1 = max(f1, value)
    return em, f1

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def load_group(path, group):
    opener = gzip.open if str(path).endswith('.gz') else open
    rows, counts, seen = [], collections.Counter(), set()
    with opener(path, 'rt', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            counts['raw_rows'] += 1
            d = json.loads(line)
            q, c = d.get('question', ''), d.get('context', '')
            answers = d.get('answers', {}).get('text', [])
            if isinstance(answers, str): answers = [answers]
            answers = [a for a in answers if isinstance(a, str) and normalise(a)]
            if not isinstance(q, str) or not isinstance(c, str) or not normalise(q) or not normalise(c) or not answers:
                counts['missing_fields_or_answers'] += 1; continue
            # Restrict this reading-comprehension study to answers supported by the supplied passage.
            answers = [a for a in answers if normalise(a) in normalise(c)]
            if not answers:
                counts['reference_not_in_context'] += 1; continue
            key = normalise(q)
            if key in seen:
                counts['repeated_question'] += 1; continue
            seen.add(key)
            cw = len(c.split())
            if not 30 <= cw <= 250:
                counts['outside_context_word_range'] += 1; continue
            at = 'number' if any(ch.isdigit() for ch in answers[0]) else ('short' if len(answers[0].split()) <= 3 else 'long')
            row = dict(group=group, id=str(d.get('id', line_number)), source_line=line_number,
                       title=d.get('title', ''), question=q, context=c, references=answers,
                       question_words=len(q.split()), context_words=cw, answer_type=at,
                       stratum=f'{cw // 50}:{at}')
            rows.append(row)
    counts['eligible_unique_questions'] = len(rows)
    return rows, dict(counts)

def prepare(args):
    root, out = Path(args.data), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != DATA_COMMIT: raise ValueError(f'Dataset revision differs: {commit}; expected {DATA_COMMIT}')
    paths = {
        'original': root / 'data/reading-comprehension/kazqad-reading-comprehension-v1.0-kk-test.jsonl',
        'translated': root / 'data/supplementary/nq-translate-kk/nq-reading-comprehension-translate-kk.jsonl.gz'}
    groups, audit = {}, {}
    for group, path in paths.items():
        groups[group], audit[group] = load_group(path, group)
        audit[group].update(source=str(path), sha256=sha(path))
    shared = {normalise(x['question']) for x in groups['original']} & {normalise(x['question']) for x in groups['translated']}
    for group in groups:
        before = len(groups[group])
        groups[group] = [x for x in groups[group] if normalise(x['question']) not in shared]
        audit[group]['cross_group_shared_questions_removed'] = before - len(groups[group])
    if args.exclude_sample:
        previous = json.loads(Path(args.exclude_sample).read_text(encoding='utf-8'))
        excluded = {normalise(x['question']) for x in previous}
        for group in groups:
            before = len(groups[group])
            groups[group] = [x for x in groups[group] if normalise(x['question']) not in excluded]
            audit[group]['previous_pilot_questions_removed'] = before - len(groups[group])
    # Match coarse passage-length and answer-type strata; topic remains an acknowledged confounder.
    by = {g: collections.defaultdict(list) for g in groups}
    for g in groups:
        for x in groups[g]: by[g][x['stratum']].append(x)
    rng = random.Random(SEED); pairs = []
    for key in sorted(set(by['original']) & set(by['translated'])):
        a, b = by['original'][key], by['translated'][key]
        rng.shuffle(a); rng.shuffle(b); pairs.extend(zip(a, b))
    rng.shuffle(pairs)
    if len(pairs) < args.n: raise ValueError(f'Only {len(pairs)} matched pairs available')
    selected = [row for pair in pairs[:args.n] for row in pair]
    dump(out / 'selected.json', selected)
    review = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k,v in x.items()} |
              {'review_topic': '', 'review_answer_supported': '', 'review_ambiguity': ''} for x in selected]
    write_csv(out / 'sample_review.csv', review)
    manifest = dict(status='DATA_PREPARED_MODEL_NOT_RUN', data_commit=commit, seed=SEED,
                    n_per_group=args.n, matching='context-word bins and answer type; no topic matching',
                    audit=audit, selected_sha256=sha(out / 'selected.json'))
    dump(out / 'preparation.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

def messages(row):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
            f"Мәтін: {row['context']}\nСұрақ: {row['question']}\nЖауап:"}]

def bootstrap(values, seed=42, B=2000):
    rng = random.Random(seed); n = len(values)
    vals = sorted(sum(rng.choices(values, k=n))/n for _ in range(B))
    return [vals[int(.025*B)], vals[min(B-1, int(.975*B))]]

def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from huggingface_hub import model_info
    out = Path(args.out)
    if (out/'predictions.json').exists(): raise FileExistsError('Use a new output directory or archive the previous predictions first')
    rows = json.loads((out/'selected.json').read_text(encoding='utf-8'))
    selected_hash = sha(out/'selected.json')
    set_seed(SEED)
    revision = model_info(MODEL, revision=args.revision).sha
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=dtype).to(device).eval()
    # Check every input before inference. Do not silently truncate away an answer or selectively drop long inputs.
    for row in rows:
        ids = tok.apply_chat_template(messages(row), add_generation_prompt=True, tokenize=True)
        if len(ids) + 64 > 4096: raise ValueError('Input exceeds predeclared 4096-token budget; revise eligibility before running either group')
    predictions=[]; start=time.time()
    for row in rows:
        inp = tok.apply_chat_template(messages(row), add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            result=model.generate(**inp, do_sample=False, max_new_tokens=64,
                                  pad_token_id=tok.eos_token_id)
        new=result[0, inp['input_ids'].shape[-1]:]
        answer=tok.decode(new, skip_special_tokens=True).strip()
        em,f1=score(answer,row['references'])
        predictions.append(row | dict(prediction=answer, exact_match=em, token_f1=f1,
                                      generated_tokens=len(new), reached_generation_limit=len(new)>=64))
        dump(out/'predictions.partial.json',predictions)
        print(row['group'],row['id'],'EM=',em,'F1=',round(f1,3),flush=True)
    dump(out/'predictions.json', predictions)
    versions={p:importlib.metadata.version(p) for p in ['torch','transformers','huggingface-hub']}
    dump(out/'run_manifest.json',dict(model=MODEL,model_revision=revision,seed=SEED,device=device,
          dtype=str(dtype),python=platform.python_version(),versions=versions,
          selected_sha256=selected_hash,system_instruction=SYSTEM,max_new_tokens=64,do_sample=False,
          runtime_seconds=time.time()-start))
    summarise(out)

def summarise(out):
    rows=json.loads((out/'predictions.json').read_text(encoding='utf-8'))
    results={}
    for group in ['original','translated']:
        sub=[r for r in rows if r['group']==group]
        results[group]={'n':len(sub)}
        for metric in ['exact_match','token_f1']:
            values=[r[metric] for r in sub]
            results[group][metric]={'mean':sum(values)/len(values),'bootstrap_ci95':bootstrap(values)}
    # These are separate source questions, not translation pairs. Resample groups independently.
    rng=random.Random(SEED); gaps={}
    for metric in ['exact_match','token_f1']:
        a=[r[metric] for r in rows if r['group']=='original']; b=[r[metric] for r in rows if r['group']=='translated']
        vals=sorted(sum(rng.choices(a,k=len(a)))/len(a)-sum(rng.choices(b,k=len(b)))/len(b) for _ in range(2000))
        gaps[metric]={'original_minus_translated':sum(a)/len(a)-sum(b)/len(b),'bootstrap_ci95':[vals[50],vals[1950]]}
    dump(out/'summary.json',dict(groups=results,gaps=gaps,interpretation='Descriptive association for this model and sample; not a causal translation effect.'))
    export=[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v) for k,v in r.items()} for r in rows]
    write_csv(out/'predictions.csv',export)
    print(json.dumps(results,ensure_ascii=False,indent=2))

def self_test():
    # Artificial strings test scoring software only; these are not research observations.
    assert score('1997.', ['1997']) == (1.0,1.0)
    assert score('қала Астана', ['Астана']) == (0.0,2/3)
    assert score('Алматы', ['Астана']) == (0.0,0.0)
    assert normalise('ҚАЗАҚ, тілі!') == 'қазақ тілі'
    print('4 scoring checks passed. No model evaluation was performed.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','run','self-test'])
    p.add_argument('--data',default='../kazqad_source'); p.add_argument('--out',default='results')
    p.add_argument('--n',type=int,default=10); p.add_argument('--revision',default='main')
    p.add_argument('--exclude-sample', default=None, help='Previous pilot selected.json to exclude before main selection')
    a=p.parse_args()
    if a.command=='prepare': prepare(a)
    elif a.command=='run': run(a)
    else: self_test()
