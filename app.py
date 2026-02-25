import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import openai
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from a video file and saves it as an audio file.
    
    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio file.
    """
    try:
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio_path}")
        print(f"Audio extracted and saved to {output_audio_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")

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
        str: Transcribed text.
    """
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        with open(audio_path, "rb") as audio_file:
            response = openai.Audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        return response["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def summarize_transcription(transcription):
    """
    Summarizes the transcription using GPT-4o.

    Args:
        transcription (str): The transcribed text to summarize.

    Returns:
        dict: A dictionary containing the structured summary with keys 'agenda', 'main_points', and 'decisions'.
    """
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        prompt = (
            "以下の会議の文字起こしを要約してください。以下のフォーマットで出力してください:\n"
            "1. 議題の説明: 会議の目的や概要\n"
            "2. 主な発言: 重要なやり取りの要約\n"
            "3. 決定事項: 確定したタスクや合意点\n"
            f"\n文字起こし:\n{transcription}"
        )

        response = openai.ChatCompletions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "あなたは会議の議事録を作成するアシスタントです。"},
                {"role": "user", "content": prompt}
            ]
        )

        summary_text = response["choices"][0]["message"]["content"]

        # Parse the summary into a structured format (basic parsing example)
        summary = {
            "agenda": "",
            "main_points": "",
            "decisions": ""
        }

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

def export_to_local_folder(summary, output_dir):
    """
    Exports the summarized data to a local folder as a text file.

    Args:
        summary (dict): The structured summary containing 'agenda', 'main_points', and 'decisions'.
        output_dir (str): Path to the directory where the file will be saved.

    Returns:
        str: Path of the created file.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, "議事録.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"議題の説明:\n{summary['agenda']}\n\n")
            file.write(f"主な発言:\n{summary['main_points']}\n\n")
            file.write(f"決定事項:\n{summary['decisions']}\n")

        return file_path
    except Exception as e:
        print(f"Error exporting to local folder: {e}")
        return None

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

    # Sidebar for API keys
    st.sidebar.header("設定")
    openai_api_key = st.sidebar.text_input("OpenAI APIキー", type="password")
    google_api_key = st.sidebar.text_input("Google APIキー", type="password")

    # Main area
    st.title("議事録作成アプリ")

    # File upload
    uploaded_file = st.file_uploader("音声または動画ファイルをアップロードしてください (mp3, m4a, mp4)", type=["mp3", "m4a", "mp4"])

    # Meeting information input
    meeting_info = st.text_area("会議情報を入力してください (サイボウズOfficeの予定情報をコピペ)")

    # Initialize session state for save button
    if "save_success" not in st.session_state:
        st.session_state.save_success = False

    if st.button("議事録生成"):
        if not uploaded_file:
            st.error("ファイルをアップロードしてください。")
            return

        if not openai_api_key:
            st.error("OpenAI APIキーを入力してください。")
            return

        # Placeholder for processing
        with st.spinner("処理中..."):
            # Save uploaded file temporarily
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Call audio extraction, transcription, and summarization functions
            audio_path = "extracted_audio.mp3"
            extract_audio_from_video(file_path, audio_path)
            transcription = transcribe_audio_with_whisper(audio_path)
            summary = summarize_transcription(transcription)

            # Display results
            if summary:
                st.subheader("生成された議事録")
                st.text_area("議題の説明", summary["agenda"], height=100)
                st.text_area("主な発言", summary["main_points"], height=200)
                st.text_area("決定事項", summary["decisions"], height=100)

                # Save button with session state
                if st.button("ローカルフォルダへ保存"):
                    output_dir = "output"
                    file_path = export_to_local_folder(summary, output_dir)
                    if file_path:
                        st.session_state.save_success = True

    # Display save success message
    if st.session_state.save_success:
        st.success("議事録がローカルフォルダに保存されました！")

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not openai_api_key or not google_api_key:
        print("APIキーが設定されていません。環境変数を確認してください。")
        exit(1)

    main()