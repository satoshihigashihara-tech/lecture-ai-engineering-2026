"""既存の回答原文をCodexが個別に確認した採点結果を記録する。LLM/APIは呼ばない。"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'lectures/02/advanced/results'

# 位置に依存しないよう、確認した質問文と判定理由を対応付ける。
ASSESSMENTS = [
    ('LoRAの概要を教えてください。',
     'LoRAの名称をLightning-Short Attention Mechanismと誤って展開。低ランク更新の説明も不足。',
     '用語集の定義と一致。元の重み固定・低ランク行列の学習を説明。', True),
    ('LoRAは元の重みをどう扱って調整しますか？',
     '名称をLightning-optimized Residual Adapterと誤展開し、残りのパラメータや残差補償の説明が参照定義と異なる。',
     '低ランク行列の積と2ビット表現を混同し、元の重み固定を説明できていない。', False),
    ('量子化の概要を教えてください。',
     '物理学の量子化を説明しており、講座のモデル重みの低ビット化に答えていない。',
     '用語集の定義と一致。少ないビット数による重み表現とメモリ削減を説明。', True),
    ('量子化では数値の表現をどう変えますか？',
     '物理量の例に終始し、講座で問う重みのビット数削減を説明していない。',
     '重みの数値表現の変更を説明せず、最小化の精度が上がるなど根拠のない説明を含む。', False),
    ('ファインチューニングの概要を教えてください。',
     'ソフトウェア全般の調整を説明し、事前学習済みモデルの追加学習を説明していない。',
     '用語集の定義と一致。特定タスク・形式に合わせた追加学習を説明。', True),
    ('追加学習はどのような目的で行うのですか？',
     '人の学習や知識活用の説明で、講座のモデル追加学習の目的に答えていない。',
     '関連語は含むが、データを追加して結果をモデルに渡して評価する手法という説明は定義と異なる。', False),
    ('トークンの概要を教えてください。',
     '暗号資産の通貨として説明し、LLMのテキスト単位に答えていない。',
     '用語集の定義と一致。テキストの単位と料金等の計数を説明。', True),
    ('tokenは何を数えるときに使う単位ですか？',
     'NLPの単位には触れるが、単語の前後の文字を除外する等の誤説明を含み、何を計数するかを十分説明していない。',
     'トークンではなくコンテキストウィンドウの定義を返している。', False),
    ('temperatureの概要を教えてください。',
     '物理的な温度を説明し、トークン選択のランダムさという意味に答えていない。',
     '用語集の定義と一致。低温で安定・高温で多様になると説明。', True),
    ('温度パラメータを上げると回答はどう変化しますか？',
     '精度が向上する、推論が深まると断定し、物理温度の説明も混ざっている。',
     '分布の均化には触れるが、多様性が増すことを明確に説明せず、参照定義にないAPI指定方法も混在するため厳格基準では不合格。', False),
    ('ハルシネーションの概要を教えてください。',
     '量子力学の相関関数という架空の説明をしており、講座の定義と異なる。',
     '用語集の定義と一致。事実と異なる情報をもっともらしく生成する現象と説明。', True),
    ('幻覚と呼ばれるLLMの誤りはどのようなものですか？',
     '因果関係の欠如の話に置き換え、誤情報をもっともらしく生成する現象を説明できていない。',
     '複数回質問して結果を返す処理と混同している。', False),
]


def main():
    path = OUT / 'result.json'
    original = OUT / 'result_colab.json'
    if not original.exists(): original.write_bytes(path.read_bytes())
    result = json.loads(original.read_text(encoding='utf-8'))
    assessments = {q: (before, after, correct) for q, before, after, correct in ASSESSMENTS}
    rows = result['detail']
    for row in rows:
        q = row['question']
        if row['group'] == 'unknown':
            # これは一般知識としての真偽を採点する指標ではない。
            row['review_criterion'] = '用語集に記載がないことを明示するか'
            row['manual_before_correct'] = False
            row['manual_after_correct'] = False
            row['manual_search_correct'] = row['search_hit'] is None
            row['before_reason'] = '用語集外と明示せず説明している。一般知識としての説明の真偽は、この判定の対象外。'
            row['after_reason'] = '用語集外と明示せず説明している。未知語への回答を控える振る舞いは確認できない。'
            row['search_reason'] = '用語集外として見つからないと返した。' if row['search_hit'] is None else '用語集外のBERTに対してLoRAのランクを誤取得した。'
        else:
            before_reason, after_reason, after_correct = assessments[q]
            row['review_criterion'] = '講座の参照定義に沿って質問に答え、重要な誤説明を含まない（厳格基準）'
            row['manual_before_correct'] = False
            row['manual_after_correct'] = after_correct
            row['manual_search_correct'] = row['search_answer'] == row['reference']
            row['before_reason'], row['after_reason'] = before_reason, after_reason
            row['search_reason'] = '参照定義を取得。' if row['manual_search_correct'] else '幻覚ではなくLLMの定義を誤取得。'
    summary = {'reviewer': 'Codexによる回答原文と用語集の照合。本人による採点を意味しない。',
               'scope': '12問の定義への適合と、用語集外3問の明示を分離して集計。一般的な性能・事実正確性の指標ではない。',
               'known': {}, 'unknown': {}, 'rows': rows}
    for label, field in [('before','manual_before_correct'),('after','manual_after_correct'),('search','manual_search_correct')]:
        for group in ['known','unknown']:
            subset = [r for r in rows if (r['group']=='unknown') == (group=='unknown')]
            summary[group][label] = {'passed':sum(r[field] for r in subset),'n':len(subset)}
    result['manual_review_status'] = 'Codexによる原文確認と採点を記入済み。採点基準・根拠はmanual_review_summary.json参照。'
    result['manual_review_summary'] = {k:v for k,v in summary.items() if k!='rows'}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'manual_review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
