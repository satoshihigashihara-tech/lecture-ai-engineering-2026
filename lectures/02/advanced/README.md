# 第2回 発展課題

資料指定のコマンドでCodespaces上で生成したglossary_sft.jsonlをColabにアップロードし、messages形式のままDataset.from_listで読み、教材のSFTTrainerで学習した。

実行用ノートブック：../../../notebooks/lec02_glossary_advanced.ipynb

実測結果はresults/、結果と考察は../answer.mdに記載。116件・20エポック・300ステップ。評価15問。実装・結果整理にCodexを使用した。

results内のJSONとCSVはColabから保存した原本を保持する。モデル重みやチェックポイントは含めない。
