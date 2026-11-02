# 第2回 発展課題

資料指定のコマンドでCodespaces上で生成したglossary_sft.jsonlをColabにアップロードし、messages形式のままDataset.from_listで読み、教材のSFTTrainerで学習した。

実行用ノートブック：../../../notebooks/lec02_glossary_advanced.ipynb

実測結果はresults/、結果と考察は../answer.mdに記載。116件・20エポック・300ステップ。評価15問。実装・結果整理にCodexを使用した。

`results/result_colab.json` と `eval_before.json`、`eval_after.json` はColabの原本を保持する。`result.json` に測定値・回答原文を保ったまま採点を追記し、`manual_review.csv` の採点欄に判定と理由を記入した。採点者はCodexであり、本人が採点したことを意味しない。基準と集計は `manual_review_summary.json` に記録する。

用語集にある語の定義への適合と、用語集にない語への非掲載の明示は異なる基準なので別集計とする。必須語句の一致率とも区別する。比較対象は簡易文字列検索であり、Geminiの検索エージェントは未実行。

モデル重みやチェックポイントは含めない。
