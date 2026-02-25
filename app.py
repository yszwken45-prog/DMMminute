import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import split_on_silence
from openai import OpenAI
import streamlit as st
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from a video file and saves it as an audio file.
    
    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio file.
    """
    try:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return False, "ffmpeg が見つかりません。PATH を確認してください。"

        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin:
            probe_command = [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                video_path,
            ]
            probe_result = subprocess.run(probe_command, capture_output=True, text=True)
            if probe_result.returncode == 0 and not probe_result.stdout.strip():
                return False, "この動画ファイルには音声トラックがありません。音声付きの mp4 または mp3/m4a をアップロードしてください。"

        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            output_audio_path,
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "ffmpeg command failed").strip()
            if "does not contain any stream" in error_message.lower():
                return False, "この動画ファイルには音声トラックがありません。音声付きの mp4 または mp3/m4a をアップロードしてください。"
            return False, error_message

        print(f"Audio extracted and saved to {output_audio_path}")
        return True, None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False, str(e)

def split_audio(audio_path, output_dir, silence_thresh=-40, min_silence_len=700):
    """
    Splits an audio file into chunks based on silence.

    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str): Directory to save the audio chunks.
        silence_thresh (int): Silence threshold in dBFS.
        min_silence_len (int): Minimum length of silence to consider for splitting (in ms).
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = split_on_silence(audio, 
                                  min_silence_len=min_silence_len, 
                                  silence_thresh=silence_thresh)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(output_dir, f"chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")
            print(f"Exported chunk {i} to {chunk_path}")
    except Exception as e:
        print(f"Error splitting audio: {e}")

def transcribe_audio_with_whisper(audio_path):
    """
    Transcribes audio using OpenAI Whisper API.

    Args:
        audio_path (str): Path to the audio file to transcribe.

    Returns:
        tuple[str | None, str | None]: (Transcribed text, error message)
    """
    try:
        client = get_openai_client()
        if not client:
            msg = "OPENAI_API_KEY is not set"
            print(f"Error during transcription: {msg}")
            return None, msg

        max_whisper_mb = 25
        file_size_bytes = os.path.getsize(audio_path)
        if file_size_bytes <= max_whisper_mb * 1024 * 1024:
            text, single_error = transcribe_single_file(client, audio_path)
            return text, single_error

        chunk_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
        try:
            chunk_paths, split_error = split_audio_for_whisper_limit(
                audio_path,
                chunk_dir,
                max_whisper_mb,
            )
            if split_error:
                print(f"Error during chunk split: {split_error}")
                return None, f"音声分割に失敗しました: {split_error}"

            transcripts = []
            for chunk_path in chunk_paths:
                chunk_text = transcribe_single_file(client, chunk_path)
                chunk_text, chunk_error = transcribe_single_file(client, chunk_path)
                if not chunk_text:
                    print(f"Error during transcription for chunk: {chunk_path}, {chunk_error}")
                    return None, f"分割音声の文字起こしに失敗しました: {chunk_error}"
                transcripts.append(chunk_text)

            return "\n".join(transcripts), None
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None, str(e)


def transcribe_single_file(client, audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text, None
    except Exception as e:
        return None, str(e)


def split_audio_for_whisper_limit(audio_path, output_dir, max_chunk_mb):
    try:
        audio = AudioSegment.from_file(audio_path)
        if len(audio) == 0:
            return [], "音声が空のため分割できません。"

        max_chunk_bytes = max_chunk_mb * 1024 * 1024
        source_size = os.path.getsize(audio_path)
        bytes_per_ms = source_size / max(len(audio), 1)
        estimated_chunk_ms = int((max_chunk_bytes * 0.9) / max(bytes_per_ms, 1e-6))
        chunk_ms = max(estimated_chunk_ms, 30_000)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        chunk_paths = []
        start_ms = 0
        chunk_index = 0

        while start_ms < len(audio):
            end_ms = min(start_ms + chunk_ms, len(audio))
            segment = audio[start_ms:end_ms]
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_index}.mp3")
            segment.export(chunk_path, format="mp3", bitrate="128k")

            while os.path.getsize(chunk_path) > max_chunk_bytes and (end_ms - start_ms) > 15_000:
                end_ms = start_ms + int((end_ms - start_ms) * 0.8)
                segment = audio[start_ms:end_ms]
                segment.export(chunk_path, format="mp3", bitrate="128k")

            if os.path.getsize(chunk_path) > max_chunk_bytes:
                return [], f"チャンク作成に失敗しました: {chunk_index}"

            chunk_paths.append(chunk_path)
            start_ms = end_ms
            chunk_index += 1

        return chunk_paths, None
    except Exception as e:
        return [], str(e)

def summarize_transcription(transcription, meeting_info=""):
    """
    Summarizes the transcription using GPT-4o.

    Args:
        transcription (str): The transcribed text to summarize.
        meeting_info (str): Cybozu schedule information pasted by the user.

    Returns:
        dict: A dictionary containing the structured summary with keys 'meeting_info', 'agenda', 'main_points', and 'decisions'.
    """
    try:
        client = get_openai_client()
        if not client:
            print("Error during summarization: OPENAI_API_KEY is not set")
            return None

        prompt = (
            "以下の会議の文字起こしを要約してください。以下のフォーマットで出力してください:\n"
            "0. 会議基本情報（必ず次の4項目をこの順で出力）:\n"
            "会議名: ...\n"
            "日時: ...\n"
            "参加者: ...\n"
            "場所/URL: ...\n"
            "1. 議題の説明: 会議の目的や概要\n"
            "2. 主な発言: 重要なやり取りの要約\n"
            "3. 決定事項: 確定したタスクや合意点\n"
            f"\nサイボウズの会議情報:\n{meeting_info or '（入力なし）'}"
            f"\n文字起こし:\n{transcription}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは会議の議事録を作成するアシスタントです。"},
                {"role": "user", "content": prompt}
            ]
        )

        summary_text = response.choices[0].message.content
        if not summary_text:
            return None

        # Parse the summary into a structured format (basic parsing example)
        fallback_meeting = parse_meeting_basic_info(meeting_info)
        summary = {
            "meeting_name": fallback_meeting["meeting_name"],
            "meeting_datetime": fallback_meeting["meeting_datetime"],
            "participants": fallback_meeting["participants"],
            "location_url": fallback_meeting["location_url"],
            "agenda": "",
            "main_points": "",
            "decisions": ""
        }

        if "0. 会議基本情報" in summary_text and "1. 議題の説明:" in summary_text:
            meeting_info_text = summary_text.split("0. 会議基本情報")[1].split("1. 議題の説明:")[0].strip()
            parsed_meeting = parse_meeting_basic_info(meeting_info_text)
            summary["meeting_name"] = parsed_meeting["meeting_name"]
            summary["meeting_datetime"] = parsed_meeting["meeting_datetime"]
            summary["participants"] = parsed_meeting["participants"]
            summary["location_url"] = parsed_meeting["location_url"]

        if "1. 議題の説明:" in summary_text:
            summary["agenda"] = summary_text.split("1. 議題の説明:")[1].split("2. 主な発言:")[0].strip()
        if "2. 主な発言:" in summary_text:
            summary["main_points"] = summary_text.split("2. 主な発言:")[1].split("3. 決定事項:")[0].strip()
        if "3. 決定事項:" in summary_text:
            summary["decisions"] = summary_text.split("3. 決定事項:")[1].strip()

        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


def parse_meeting_basic_info(text):
    default_info = {
        "meeting_name": "不明",
        "meeting_datetime": "不明",
        "participants": "不明",
        "location_url": "不明",
    }

    if not text:
        return default_info

    patterns = {
        "meeting_name": r"会議名\s*[:：]\s*(.+)",
        "meeting_datetime": r"日時\s*[:：]\s*(.+)",
        "participants": r"参加者\s*[:：]\s*(.+)",
        "location_url": r"場所\s*/\s*URL\s*[:：]\s*(.+)",
    }

    parsed = default_info.copy()
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match and match.group(1).strip():
            parsed[key] = match.group(1).strip()

    return parsed

def export_to_local_folder(summary, output_dir):
    """
    Exports the summarized data to a local folder as a text file.

    Args:
        summary (dict): The structured summary containing 'agenda', 'main_points', and 'decisions'.
        output_dir (str): Path to the directory where the file will be saved.

    Returns:
        tuple[str | None, str | None]: (Path of the created file, error message)
    """
    try:
        output_dir_abs = os.path.abspath(output_dir)
        if not os.path.exists(output_dir_abs):
            os.makedirs(output_dir_abs)

        file_path = os.path.join(output_dir_abs, "議事録.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(build_minutes_text(summary))

        return file_path, None
    except Exception as e:
        print(f"Error exporting to local folder: {e}")
        return None, str(e)


def export_transcription_to_local_folder(transcription, output_dir):
    """
    Exports the raw transcription text to a local folder as a text file.

    Args:
        transcription (str): Raw transcription text.
        output_dir (str): Path to the directory where the file will be saved.

    Returns:
        tuple[str | None, str | None]: (Path of the created file, error message)
    """
    try:
        output_dir_abs = os.path.abspath(output_dir)
        if not os.path.exists(output_dir_abs):
            os.makedirs(output_dir_abs)

        file_path = os.path.join(output_dir_abs, "文字起こし生データ.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(build_transcription_raw_text(transcription))

        return file_path, None
    except Exception as e:
        print(f"Error exporting transcription to local folder: {e}")
        return None, str(e)


def build_transcription_raw_text(transcription):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"生成日時: {generated_at}\n"
        "\n"
        "--- 文字起こし生データ ---\n"
        f"{transcription or ''}\n"
    )


def build_minutes_text(summary):
    return (
        "会議基本情報:\n"
        f"会議名: {summary.get('meeting_name', '不明')}\n"
        f"日時: {summary.get('meeting_datetime', '不明')}\n"
        f"参加者: {summary.get('participants', '不明')}\n"
        f"場所/URL: {summary.get('location_url', '不明')}\n\n"
        f"議題の説明:\n{summary['agenda']}\n\n"
        f"主な発言:\n{summary['main_points']}\n\n"
        f"決定事項:\n{summary['decisions']}\n"
    )

def cleanup_old_files(directory, retention_period_days=90):
    """
    Deletes files older than the specified retention period from the given directory.

    Args:
        directory (str): Path to the directory to clean up.
        retention_period_days (int): Number of days to retain files. Files older than this will be deleted.
    """
    try:
        current_time = time.time()
        retention_period_seconds = retention_period_days * 24 * 66 * 60 * 60

        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > retention_period_seconds:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def process_vtt_file(vtt_path):
    """
    Processes a .vtt file and extracts text content.

    Args:
        vtt_path (str): Path to the .vtt file.

    Returns:
        str: Extracted text content.
    """
    try:
        with open(vtt_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Filter out metadata and timestamps, keep only text lines
        text_lines = []
        for line in lines:
            line = line.strip()
            if not line or "-->" in line or line.isdigit():
                continue
            text_lines.append(line)

        return "\n".join(text_lines)
    except Exception as e:
        print(f"Error processing .vtt file: {e}")
        return None

def main():
    """
    Streamlit app for file upload and transcription.
    """
    st.set_page_config(page_title="議事録作成アプリ", layout="wide")

    # Main area
    st.title("議事録作成アプリ")

    # Initialize session state
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "save_success" not in st.session_state:
        st.session_state.save_success = False
    if "saved_file_path" not in st.session_state:
        st.session_state.saved_file_path = None
    if "save_error" not in st.session_state:
        st.session_state.save_error = None
    if "meeting_info_input" not in st.session_state:
        st.session_state.meeting_info_input = ""
    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0
    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "transcription_save_success" not in st.session_state:
        st.session_state.transcription_save_success = False
    if "saved_transcription_file_path" not in st.session_state:
        st.session_state.saved_transcription_file_path = None
    if "transcription_save_error" not in st.session_state:
        st.session_state.transcription_save_error = None

    # File upload
    size_limit_label = st.radio(
        "最大ファイルサイズ",
        options=["25MB", "75MB"],
        horizontal=True,
        help="75MBを選ぶと、Whisper制限（25MB）を超える部分は自動分割して処理します。",
    )
    max_upload_mb = 75 if size_limit_label == "75MB" else 25

    uploaded_file = st.file_uploader(
        "音声または動画ファイルをアップロードしてください (mp3, m4a, mp4)",
        type=["mp3", "m4a", "mp4"],
        key=f"uploaded_file_{st.session_state.uploader_version}",
    )

    # Meeting information input
    meeting_info = st.text_area(
        "会議情報を入力してください (サイボウズOfficeの予定情報をコピペ)",
        key="meeting_info_input"
    )

    left_col, right_col = st.columns(2)
    generate_clicked = left_col.button("議事録生成")
    clear_clicked = right_col.button("クリア")

    if clear_clicked:
        st.session_state.summary = None
        st.session_state.transcription = None
        st.session_state.save_success = False
        st.session_state.saved_file_path = None
        st.session_state.save_error = None
        st.session_state.transcription_save_success = False
        st.session_state.saved_transcription_file_path = None
        st.session_state.transcription_save_error = None
        st.session_state.meeting_info_input = ""
        st.session_state.uploader_version += 1
        st.rerun()

    if generate_clicked:
        if not uploaded_file:
            st.error("ファイルをアップロードしてください。")
        elif uploaded_file.size > max_upload_mb * 1024 * 1024:
            st.error(f"ファイルサイズが上限を超えています（選択上限: {max_upload_mb}MB）。")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY が設定されていません。.env を確認してください。")
        else:
            # Placeholder for processing
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

    # Display results if they exist in session state
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
            "議事録をPCにダウンロード",
            data=build_minutes_text(st.session_state.summary),
            file_name="議事録.txt",
            mime="text/plain",
        )

        if st.session_state.transcription:
            st.download_button(
                "文字起こし生データをPCにダウンロード",
                data=build_transcription_raw_text(st.session_state.transcription),
                file_name="文字起こし生データ.txt",
                mime="text/plain",
            )

        # Save button with session state
        if st.button("ローカルフォルダへ保存"):
            output_dir = "output"
            file_path, save_error = export_to_local_folder(st.session_state.summary, output_dir)
            if file_path:
                st.session_state.save_success = True
                st.session_state.saved_file_path = file_path
                st.session_state.save_error = None
            else:
                st.session_state.save_success = False
                st.session_state.saved_file_path = None
                st.session_state.save_error = save_error

        if st.session_state.transcription and st.button("文字起こし生データをローカルフォルダへ保存"):
            output_dir = "output"
            file_path, save_error = export_transcription_to_local_folder(st.session_state.transcription, output_dir)
            if file_path:
                st.session_state.transcription_save_success = True
                st.session_state.saved_transcription_file_path = file_path
                st.session_state.transcription_save_error = None
            else:
                st.session_state.transcription_save_success = False
                st.session_state.saved_transcription_file_path = None
                st.session_state.transcription_save_error = save_error

    # Display save success message
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