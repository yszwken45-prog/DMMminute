import os
import uuid

import streamlit as st

from constants import (
    APP_TITLE,
    BUTTON_CLEAR,
    BUTTON_GENERATE,
    BUTTON_SAVE_MINUTES,
    BUTTON_SAVE_RAW,
    DOWNLOAD_MINUTES_LABEL,
    DOWNLOAD_RAW_LABEL,
    MAX_UPLOAD_MB_BY_LABEL,
    MEETING_INFO_PROMPT,
    OUTPUT_DIR,
    PAGE_LAYOUT,
    SIZE_LIMIT_OPTIONS,
    UPLOAD_PROMPT,
    UPLOAD_TYPES,
)
from function import (
    build_minutes_text,
    build_transcription_raw_text,
    clear_session_state,
    export_to_local_folder,
    export_transcription_to_local_folder,
    extract_audio_from_video,
    initialize_session_state,
    summarize_transcription,
    transcribe_audio_with_whisper,
)

def main():
    st.set_page_config(page_title=APP_TITLE, layout=PAGE_LAYOUT)

    st.title(APP_TITLE)

    initialize_session_state()

    size_limit_label = st.radio(
        "最大ファイルサイズ",
        options=SIZE_LIMIT_OPTIONS,
        horizontal=True,
        help="75MBを選ぶと、Whisper制限（25MB）を超える部分は自動分割して処理します。",
    )
    max_upload_mb = MAX_UPLOAD_MB_BY_LABEL[size_limit_label]

    uploaded_file = st.file_uploader(
        UPLOAD_PROMPT,
        type=UPLOAD_TYPES,
        key=f"uploaded_file_{st.session_state.uploader_version}",
    )

    meeting_info = st.text_area(
        MEETING_INFO_PROMPT,
        key="meeting_info_input",
    )

    left_col, right_col = st.columns(2)
    generate_clicked = left_col.button(BUTTON_GENERATE)
    clear_clicked = right_col.button(BUTTON_CLEAR)

    if clear_clicked:
        clear_session_state()
        st.rerun()

    if generate_clicked:
        if not uploaded_file:
            st.error("ファイルをアップロードしてください。")
        elif uploaded_file.size > max_upload_mb * 1024 * 1024:
            st.error(f"ファイルサイズが上限を超えています（選択上限: {max_upload_mb}MB）。")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY が設定されていません。.env を確認してください。")
        else:
            with st.spinner("処理中..."):
                temp_files = []
                try:
                    safe_name = os.path.basename(uploaded_file.name)
                    file_path = f"temp_{uuid.uuid4().hex}_{safe_name}"
                    temp_files.append(file_path)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    _, ext = os.path.splitext(uploaded_file.name.lower())
                    if ext == ".mp4":
                        audio_path = f"extracted_{uuid.uuid4().hex}.mp3"
                        temp_files.append(audio_path)
                        extracted, extract_error = extract_audio_from_video(file_path, audio_path)
                        if not extracted:
                            st.error(f"音声抽出に失敗しました: {extract_error}")
                            st.stop()
                    else:
                        audio_path = file_path

                    transcription, transcription_error = transcribe_audio_with_whisper(audio_path)
                    if not transcription:
                        st.error(
                            "文字起こしに失敗しました。"
                            f"\n詳細: {transcription_error or '不明なエラー'}"
                        )
                        st.stop()

                    st.session_state.transcription = transcription

                    st.session_state.summary = summarize_transcription(transcription, meeting_info)
                    if not st.session_state.summary:
                        st.error("要約に失敗しました。しばらくして再試行してください。")
                        st.stop()

                    st.session_state.save_success = False
                finally:
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

    if st.session_state.summary:
        st.subheader("生成された議事録")
        st.text_input("会議名", st.session_state.summary.get("meeting_name", "不明"), disabled=True)
        st.text_input("日時", st.session_state.summary.get("meeting_datetime", "不明"), disabled=True)
        st.text_area("参加者", st.session_state.summary.get("participants", "不明"), height=80, disabled=True)
        st.text_input("場所/URL", st.session_state.summary.get("location_url", "不明"), disabled=True)
        st.text_area("議題の説明", st.session_state.summary["agenda"], height=100)
        st.text_area("主な発言", st.session_state.summary["main_points"], height=200)
        st.text_area("決定事項", st.session_state.summary["decisions"], height=100)

        st.download_button(
            DOWNLOAD_MINUTES_LABEL,
            data=build_minutes_text(st.session_state.summary),
            file_name="議事録.txt",
            mime="text/plain",
        )

        if st.session_state.transcription:
            st.download_button(
                DOWNLOAD_RAW_LABEL,
                data=build_transcription_raw_text(st.session_state.transcription),
                file_name="文字起こし生データ.txt",
                mime="text/plain",
            )

        if st.button(BUTTON_SAVE_MINUTES):
            file_path, save_error = export_to_local_folder(st.session_state.summary, OUTPUT_DIR)
            if file_path:
                st.session_state.save_success = True
                st.session_state.saved_file_path = file_path
                st.session_state.save_error = None
            else:
                st.session_state.save_success = False
                st.session_state.saved_file_path = None
                st.session_state.save_error = save_error

        if st.session_state.transcription and st.button(BUTTON_SAVE_RAW):
            file_path, save_error = export_transcription_to_local_folder(st.session_state.transcription, OUTPUT_DIR)
            if file_path:
                st.session_state.transcription_save_success = True
                st.session_state.saved_transcription_file_path = file_path
                st.session_state.transcription_save_error = None
            else:
                st.session_state.transcription_save_success = False
                st.session_state.saved_transcription_file_path = None
                st.session_state.transcription_save_error = save_error

    if st.session_state.save_success:
        st.success(f"議事録がローカルフォルダに保存されました！\n保存先: {st.session_state.saved_file_path}")
    elif st.session_state.save_error:
        st.error(f"保存に失敗しました: {st.session_state.save_error}")

    if st.session_state.transcription_save_success:
        st.success(
            "文字起こし生データがローカルフォルダに保存されました！"
            f"\n保存先: {st.session_state.saved_transcription_file_path}"
        )
    elif st.session_state.transcription_save_error:
        st.error(f"文字起こし生データの保存に失敗しました: {st.session_state.transcription_save_error}")

if __name__ == "__main__":
    main()