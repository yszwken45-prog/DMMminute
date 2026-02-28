APP_TITLE = "議事録作成アプリ"
PAGE_LAYOUT = "wide"

SIZE_LIMIT_OPTIONS = ["25MB", "75MB"]
MAX_UPLOAD_MB_BY_LABEL = {
    "25MB": 25,
    "75MB": 75,
}
UPLOAD_TYPES = ["mp3", "m4a", "mp4"]

UPLOAD_PROMPT = "音声または動画ファイルをアップロードしてください (mp3, m4a, mp4)"
MEETING_INFO_PROMPT = "会議情報を入力してください (サイボウズOfficeの予定情報をコピペ)"
PPTX_UPLOAD_PROMPT = "参考資料（PowerPoint）をアップロードしてください（任意）"

BUTTON_GENERATE = "議事録生成"
BUTTON_CLEAR = "クリア"
BUTTON_SAVE_MINUTES = "ローカルフォルダへ保存"
BUTTON_SAVE_RAW = "文字起こし生データをローカルフォルダへ保存"

DOWNLOAD_MINUTES_LABEL = "議事録をPCにダウンロード"
DOWNLOAD_RAW_LABEL = "文字起こし生データをPCにダウンロード"

MINUTES_FILE_NAME = "議事録.txt"
RAW_TRANSCRIPTION_FILE_NAME = "文字起こし生データ.txt"
OUTPUT_DIR = "output"

OPENAI_WHISPER_MODEL = "whisper-1"
OPENAI_SUMMARY_MODEL = "gpt-4o-mini"
WHISPER_MAX_FILE_MB = 25

# 言語指定: 日本語に固定することで認識精度を向上
WHISPER_LANGUAGE = "ja"

# 文字起こし後の単語置換辞書
# キー: Whisperが誤認識しやすい表記 → 値: 正しい表記
# 自社の専門用語・固有名詞に合わせて追加・編集してください
TRANSCRIPTION_WORD_REPLACEMENTS = {
    "配客状況": "廃却状況","政策省":"製作所"
    # 例: "かいぎ": "会議",
    # 例: "えんじにあ": "エンジニア",
}

# プロンプトヒント: 会議・議事録でよく使われる語句を渡すことで固有名詞や専門用語の精度を向上
WHISPER_PROMPT = (
    "これは日本語の会議音声です。議事録として文字起こしを行います。"
    "参加者の発言をできるだけ正確に文字起こしてください。"
    "アジェンダ、決定事項、タスク、期限、担当者などの情報を正確に記録してください。"
)

OPENAI_SYSTEM_PROMPT = "あなたは会議の議事録を作成するアシスタントです。"
SUMMARY_PROMPT_TEMPLATE = (
    "以下の会議の文字起こしを要約してください。以下のフォーマットで出力してください:\n"
    "0. 会議基本情報（必ず次の4項目をこの順で出力）:\n"
    "会議名: ...\n"
    "日時: ...\n"
    "参加者: ...\n"
    "場所/URL: ...\n"
    "1. 議題の説明: 会議の目的や概要\n"
    "2. 主な発言: 重要なやり取りの要約（数値を伴う発言を優先）\n"
    "3. 決定事項: 確定したタスクや合意点\n"
    "\n出力ルール:\n"
    "- 2. 主な発言 には、件数・金額・割合・日付・時刻・期限・回数などの数値を含む発言を優先して記載してください。\n"
    "- 可能であれば数値をそのまま残してください（例: 3件、15%、2026/03/10、30分）。\n"
    "- 数値を伴う発言がない場合のみ、通常の重要発言を記載してください。\n"
    "\nサイボウズの会議情報:\n{meeting_info}"
    "\n参考資料（PowerPoint）の内容:\n{reference_material}"
    "\n文字起こし:\n{transcription}"
)

SESSION_DEFAULTS = {
    "summary": None,
    "save_success": False,
    "saved_file_path": None,
    "save_error": None,
    "meeting_info_input": "",
    "uploader_version": 0,
    "transcription": None,
    "transcription_save_success": False,
    "saved_transcription_file_path": None,
    "transcription_save_error": None,
}
