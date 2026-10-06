"""本データ（合成対話）の生成パイプライン。各モジュールの「data_spec §n」は docs/ja-data-spec.md の節を指す。

段: scripts_llm（台本）-> tts（発話ごとの音声）-> align（単語時刻と品質検査）-> assemble（対話の組み立て）
-> japersonaplex.prepare（npz）。通しは scripts/datagen.sh。使い方と結果は docs/ja-datagen.md。
"""
