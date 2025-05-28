import streamlit as st
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import json
import sys
import traceback

PROJECT_ROOT_APP = Path(__file__).resolve().parent
if str(PROJECT_ROOT_APP) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_APP))

try:
    from src.utils.model_saver import ModelSaver
    from src.embeddings.clap_embed import CLAPEmbedder
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


class AppConfig:
    def __init__(self, project_root):
        self.PROJECT_ROOT = project_root
        self.MODEL_PATH = self.PROJECT_ROOT / "models" / "catboost_model.cbm"
        self.CLAP_CHECKPOINT_PATH_STR = "models/music_speech_epoch_15_esc_89.25.pt"
        self.CLAP_CHECKPOINT_FULL_PATH_CHECK = self.PROJECT_ROOT / self.CLAP_CHECKPOINT_PATH_STR


class AlbumDataProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.catboost_model = None

    def _save_uploaded_files(self, uploaded_files, artist_name, album_name, temp_dir):
        if not artist_name or not album_name:
            return []

        artist_dir = "".join(c if c.isalnum() or c in " -" else "_" for c in artist_name)
        album_dir = "".join(c if c.isalnum() or c in " -" else "_" for c in album_name)
        album_path = temp_dir / artist_dir / album_dir
        album_path.mkdir(parents=True, exist_ok=True)

        paths = []
        for file in uploaded_files:
            path = album_path / file.name
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            paths.append(str(path.resolve()))
        return paths

    def _generate_embeddings(self, embedder: CLAPEmbedder, paths: list, temp_dir: Path):
        if not paths:
            return []
        original_dir = embedder.music_dir
        embedder.music_dir = str(temp_dir)
        data = embedder.process_files(paths)
        embedder.music_dir = original_dir
        return data

    def _prepare_features(self, song_data: list):
        if not song_data:
            raise ValueError("No song data.")

        audio_embeddings = np.array([d["audio_embedding"] for d in song_data])
        if audio_embeddings.size == 0:
            raise ValueError("Empty audio embeddings.")
        mean_audio = np.mean(audio_embeddings, axis=0)

        text_embeddings = [d.get("text_embedding") for d in song_data if d.get("text_embedding") is not None]
        if not text_embeddings:
            mean_text = np.zeros_like(mean_audio)
        else:
            mean_text = np.mean(text_embeddings, axis=0)

        return np.concatenate([mean_audio, mean_text]).reshape(1, -1)

    def _load_model(self):
        if self.catboost_model is None:
            if not self.config.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {self.config.MODEL_PATH}")
            self.catboost_model = ModelSaver.load(self.config.MODEL_PATH)
        return self.catboost_model

    def predict(self, features: np.ndarray):
        model = self._load_model()
        return model.predict(features)[0]

    def process(self, embedder: CLAPEmbedder, uploaded_files, artist, album):
        with tempfile.TemporaryDirectory(prefix="album_eval_") as temp_dir:
            temp_path = Path(temp_dir)
            paths = self._save_uploaded_files(uploaded_files, artist, album, temp_path)
            if not paths:
                raise ValueError("No files saved.")
            st.write(f"Saved {len(paths)} songs.")

            song_data = self._generate_embeddings(embedder, paths, temp_path)
            if not song_data:
                raise ValueError("No embeddings generated.")
            st.success(f"Generated embeddings for {len(song_data)} songs.")

            features = self._prepare_features(song_data)
            score = self.predict(features)
            summary = {
                "artist": artist,
                "album": album,
                "songs": len(song_data),
                "sample": song_data[0],
                "features_shape": features.shape
            }
            return score, summary


class AlbumEvaluatorApp:
    def __init__(self):
        self.config = AppConfig(PROJECT_ROOT_APP)
        self.clap_embedder = None
        self.processor = AlbumDataProcessor(self.config)

    @st.cache_resource
    def _init_embedder(_self):
        st.write("Loading CLAP model...")
        try:
            embedder = CLAPEmbedder(
                music_dir=tempfile.gettempdir(),
                checkpoint_path=_self.config.CLAP_CHECKPOINT_PATH_STR,
                output_file=str(Path(tempfile.gettempdir()) / "clap_out.json"),
                batch_size=4
            )
            embedder.load_model()
            st.success("CLAP model loaded.")
            return embedder
        except Exception as e:
            st.error(f"CLAP init failed: {e}")
            st.text(traceback.format_exc())
            return None

    def _sidebar(self):
        st.sidebar.header("Model Info")
        st.sidebar.info(f"CLAP: `{self.config.CLAP_CHECKPOINT_PATH_STR}`")
        st.sidebar.info(f"CatBoost: `{self.config.MODEL_PATH.name}`")

    def _validate(self, artist, album, files):
        if not artist or not album:
            st.error("Provide artist and album name.")
            return False
        if not files:
            st.error("Upload song files.")
            return False
        if not self.config.MODEL_PATH.exists():
            st.error("CatBoost model missing.")
            return False
        if not self.config.CLAP_CHECKPOINT_FULL_PATH_CHECK.exists():
            st.error("CLAP checkpoint missing.")
            return False
        if self.clap_embedder is None:
            st.error("CLAP model not loaded.")
            return False
        return True

    def run(self):
        st.set_page_config(layout="wide")
        st.title("🎵 Album Evaluator 🎵")

        if self.clap_embedder is None:
            self.clap_embedder = self._init_embedder()

        self._sidebar()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Album Details")
            artist = st.text_input("Artist Name", placeholder="e.g. Kendrick Lamar")
            album = st.text_input("Album Name", placeholder="e.g. DAMN.")
            files = st.file_uploader(
                "Upload Songs",
                type=["mp3", "wav", "flac", "ogg", "m4a"],
                accept_multiple_files=True
            )

        if st.button("✨ Evaluate Album ✨", use_container_width=True):
            if self._validate(artist, album, files):
                with st.spinner("Evaluating album..."):
                    try:
                        score, summary = self.processor.process(self.clap_embedder, files, artist, album)
                        with col2:
                            st.subheader("📈 Score")
                            st.metric(f"{album} by {artist}", f"{score:.2f}")
                            with st.expander("Details"):
                                st.json(summary)
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.text(traceback.format_exc())
