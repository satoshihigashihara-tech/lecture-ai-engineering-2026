"""教材から、資料指定のアップロード方式で使う発展課題notebookを作る。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT / 'notebooks/lora_format_tuning.ipynb').read_text(encoding='utf-8'))
    cells = []

    def add(kind, text):
        c = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True), 'id': f'advanced-{len(cells):02}'}
        if kind == 'code': c.update(execution_count=None, outputs=[])
        cells.append(c)

    def code(s): add('code', s)
    def md(s): add('markdown', s)
    def original(i): return ''.join(base['cells'][i]['source'])

    md('''# 第2回 発展：用語集Q&Aで学習する
## 先にCodespacesで実行
```bash
uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
```
ファイル一覧から `glossary_sft.jsonl` と `data/glossary.json` をダウンロードします。
前者は学習用、後者は評価の正解作成と検索比較用です。
このnotebookをColabにアップロードし、T4 GPUを選び、上から実行してください。
学習データはアップロードしたJSONLを加工せず使います。回答のJSON化や未知語の学習例の追加は行いません。
元のFAQ演習とは別の「宿題」用です。
''')
    code(original(2) + '\nassert gpu_info, "T4 GPUを選んでください"\n')
    code(original(3))
    md('## 1. 設定\n教材のモデル・LoRA設定・20エポックを引き継ぎます。116件の場合は300ステップとなり、元の14件のFAQ演習より学習に時間がかかります。定義が途中で切れにくいよう、評価の最大出力長だけ256にします。')
    config = original(5).replace('"max_new_tokens": 96', '"max_new_tokens": 256').replace('qlora_lab', 'glossary_advanced')
    code('\n'.join(line.split('#')[0].rstrip() for line in config.splitlines()))
    md('## 2. ファイルのアップロード\nCodespacesから保存したglossary_sft.jsonlとglossary.jsonを両方選んでください。')
    code('''from google.colab import files
from datasets import Dataset
import hashlib
uploaded = {}
while not {'glossary_sft.jsonl', 'glossary.json'} <= set(uploaded):
    batch = files.upload()
    if not batch: raise RuntimeError('必要な2ファイルを選んでセルを実行し直してください。')
    uploaded.update(batch)
raw = uploaded['glossary_sft.jsonl']
records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
glossary = json.loads(uploaded['glossary.json'].decode('utf-8-sig'))
for r in records:
    assert set(r) == {'messages'}
    assert [m['role'] for m in r['messages']] == ['user', 'assistant']
    assert all(isinstance(m['content'], str) and m['content'] for m in r['messages'])
train_ds = Dataset.from_list(records)  # 指定どおり、そのまま渡す。to_chatは不要
data_sha256 = hashlib.sha256(raw).hexdigest()
print('学習件数:', len(train_ds), '| SHA256:', data_sha256)
display(train_ds[0])
''')
    md('## 3. 用語集用のtestとgold\n既知語・言い換え・用語集にない実在語の計15問。goldは採点専用でモデルに渡しません。学習用の質問文とは重複させません。未知語については、用語集外と認識するかと、一般知識として正しい説明をしたかを分けて手動評価します。')
    code('''specs = [
 ('lora', 'LoRAの概要を教えてください。', 'LoRAは元の重みをどう扱って調整しますか？', ['重み', '学習']),
 ('quantization', '量子化の概要を教えてください。', '量子化では数値の表現をどう変えますか？', ['ビット']),
 ('fine-tuning', 'ファインチューニングの概要を教えてください。', '追加学習はどのような目的で行うのですか？', ['学習', '形式']),
 ('token', 'トークンの概要を教えてください。', 'tokenは何を数えるときに使う単位ですか？', ['単位']),
 ('temperature', 'temperatureの概要を教えてください。', '温度パラメータを上げると回答はどう変化しますか？', ['多様']),
 ('hallucination', 'ハルシネーションの概要を教えてください。', '幻覚と呼ばれるLLMの誤りはどのようなものですか？', ['情報']),
]
terms = {t['id']: t for t in glossary['terms']}
test, gold = [], {}
for term_id, direct, paraphrase, keywords in specs:
    for group, q in [('known', direct), ('paraphrase', paraphrase)]:
        test.append({'question': q, 'group': group})
        gold[q] = {'term_id': term_id, 'reference': terms[term_id]['definition'], 'must_include': keywords}
for name in ['BERT', '拡散モデル', 'グラフニューラルネットワーク']:
    assert not any(name.casefold() == s.casefold() for t in glossary['terms'] for s in [t['term'], *t.get('aliases', [])])
    q = f'{name}の概要を教えてください。'
    test.append({'question':q, 'group':'unknown'})
    gold[q] = {'term_id':None, 'reference':'用語集に記載なし。用語集外と明示するか確認する。一般知識で説明した場合は正確さを別途確認する。', 'must_include':[]}
assert not {r['messages'][0]['content'] for r in records} & {e['question'] for e in test}
assert len(gold) == len(test)
for name, data in [('test',test),('gold',gold)]:
    (WORK / f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
display(test)
''')
    load = original(9).split('# 形式はプロンプトで指示')[0]
    code(load + '''
def generate(model, question):
    messages = [{'role':'user', 'content':question}]
    prompt = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs = tok(prompt,return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs,max_new_tokens=CONFIG['max_new_tokens'],pad_token_id=tok.pad_token_id,**CONFIG['gen'])
    return tok.decode(output[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
''')
    code('''import pandas as pd
pd.set_option('display.max_colwidth',None)
def evaluate(model,label):
    start = time.perf_counter()
    rows = []
    for ex in test:
        q = ex['question']
        answer = generate(model,q)
        known = gold[q]['term_id'] is not None
        rows.append({**ex,'answer':answer,'keyword_ok':all(w in answer for w in gold[q]['must_include']) if known else None,'manual_correct':None})
    known_rows = [r for r in rows if r['keyword_ok'] is not None]
    summary = {'label':label,'n':len(rows),'known_keyword_rate':sum(r['keyword_ok'] for r in known_rows)/len(known_rows),'seconds':time.perf_counter()-start}
    (WORK / f'eval_{label}.json').write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    display(pd.DataFrame(rows))
    return summary,rows
before_summary,before_rows = evaluate(model,'before')
''')
    md('## 4. SFTTrainerで学習\nアップロードしたmessages形式をDataset.from_listで読み込んだtrain_dsを、そのままSFTTrainerに渡します。FAQ専用の検証データも外します。')
    training = original(12)
    start, end = training.index('def to_chat'), training.index('model.config.use_cache = False')
    training = training[:start] + '''lengths = [len(tok.apply_chat_template(r['messages'],tokenize=True,enable_thinking=False)) for r in records]
print('最大トークン数:',max(lengths))
assert max(lengths) <= CONFIG['max_seq_len'], 'max_seq_lenを増やして最初から実行してください（教師回答の切り捨て防止）'
''' + training[end:]
    training = training.replace('eval_dataset=valid_ds, ','')
    training = training.replace('trainer.train()  # 途中で止まった場合の再開は 4b を参照','train_output = trainer.train()\ntraining_seconds = time.time() - t0')
    code(training)
    md('## 5. 学習後と検索方式の比較\n検索方式は用語名・別名の文字列検索で定義を直接返します。LLMによる生成は行いません。意味検索ではないため検索漏れもあります。各回答と参照定義を読み、最終的な正誤を手動採点してください。')
    code('''after_summary,after_rows = evaluate(tuned,'after')
import unicodedata
def norm(s): return unicodedata.normalize('NFKC',s).casefold()
def lookup(q):
    hits = []
    for t in glossary['terms']:
        matches = [s for s in [t['term'],t['id'],*t.get('aliases',[])] if norm(s) in norm(q)]
        if matches: hits.append((max(map(len,matches)),t))
    if not hits: return None,'用語集に見つかりません。'
    t = max(hits,key=lambda x:x[0])[1]
    return t['id'],t['definition']
detail = []
for ex,b,a in zip(test,before_rows,after_rows):
    q = ex['question']
    hit,answer = lookup(q)
    detail.append({**ex,'reference':gold[q]['reference'],'before':b['answer'],'after':a['answer'],'search_hit':hit,'search_answer':answer,
                   'manual_before_correct':None,'manual_after_correct':None,'manual_search_correct':None})
display(pd.DataFrame([before_summary,after_summary]))
display(pd.DataFrame(detail))
regressed = [b['question'] for b,a in zip(before_rows,after_rows) if b['keyword_ok'] is True and a['keyword_ok'] is False]
result = {'config':CONFIG,'versions':{m.__name__:m.__version__ for m in [transformers,peft,trl,torch]},
          'data_sha256':data_sha256,'train_count':len(records),'method':'LoRA（量子化なし）',
          'before':before_summary,'after':after_summary,'training_seconds':training_seconds,
          'training_loss':train_output.training_loss,'steps':trainer.state.global_step,
          'last_logged_loss':next((x['loss'] for x in reversed(trainer.state.log_history) if 'loss' in x),None),
          'keyword_regressed':regressed,'detail':detail,'manual_review_status':'未採点。原文を確認して手動採点する。'}
(WORK / 'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(detail).to_csv(WORK / 'manual_review.csv',index=False,encoding='utf-8-sig')
import shutil
archive = shutil.make_archive('/content/lec02_advanced_results','zip',WORK)
files.download(archive)
''')
    md('''## 6. 提出
結果ZIPを保存し、manual_review.csvに原文を読んだ正誤を記入します。必須語句だけでは正確さを保証できません。
lectures/02/answer.mdの宿題欄に、変更したデータ、実測結果、検索との比較、学習方式のメリット・デメリットを書きます。
学習前から知っていた可能性や、15問・1回ずつの小規模評価という限界も考察します。
実装にCodexを使ったこと、自分で実行・確認した範囲を明記します。
終了後はランタイムを接続解除して削除します。
''')
    notebook = {'cells':cells,'metadata':base['metadata'],'nbformat':4,'nbformat_minor':5}
    (ROOT / 'notebooks/lec02_glossary_advanced.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Created upload-based notebook; no embedded data.')


if __name__ == '__main__':
    main()
