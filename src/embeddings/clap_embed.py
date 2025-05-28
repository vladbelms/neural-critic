import torch
import os
from pathlib import Path
import json
import numpy as np
import laion_clap

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent


class CLAPEmbedder:
    DEFAULT_CHECKPOINT_FILENAME = "music_speech_epoch_15_esc_89.25.pt"
    DEFAULT_CHECKPOINT_URL = (
        "https://huggingface.co/lukewys/laion_clap/resolve/main/"
        "music_speech_epoch_15_esc_89.25.pt?download=true"
    )

    def __init__(self, music_dir, checkpoint_path, output_file, batch_size=16):
        self.music_dir = Path(music_dir)
        if not self.music_dir.is_absolute():
            self.music_dir = (PROJECT_ROOT / self.music_dir).resolve()

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = (PROJECT_ROOT / self.checkpoint_path).resolve()

        self.output_file = Path(output_file)
        if not self.output_file.is_absolute():
            self.output_file = (PROJECT_ROOT / self.output_file).resolve()

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = None
        self.audio_file_paths = []
        self.embeddings_data = []

    def _download_checkpoint(self, destination_path: Path):
        if destination_path.name != self.DEFAULT_CHECKPOINT_FILENAME:
            raise FileNotFoundError(
                f"Checkpoint file '{destination_path}' not found. "
                f"Automatic download is only configured for '{self.DEFAULT_CHECKPOINT_FILENAME}'."
            )

        if requests is None:
            print("ERROR: The 'requests' library is required to download the checkpoint.")
            raise ImportError("Missing 'requests' library.")

        print(f"Attempting to download checkpoint from {self.DEFAULT_CHECKPOINT_URL}...")

        try:
            response = requests.get(self.DEFAULT_CHECKPOINT_URL, stream=True, allow_redirects=True)
            response.raise_for_status()

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 8192

            progress_bar = None
            if tqdm and total_size_in_bytes > 0:
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True,
                                    desc=f"Downloading {destination_path.name}")
            elif total_size_in_bytes > 0:
                print(f"Downloading {destination_path.name} ({total_size_in_bytes / (1024 * 1024):.2f} MB)...")
            else:
                print(f"Downloading {destination_path.name} (size unknown)...")

            with open(destination_path, 'wb') as file:
                for data in response.iter_content(chunk_size=block_size):
                    if progress_bar:
                        progress_bar.update(len(data))
                    file.write(data)

            if progress_bar:
                progress_bar.close()
                if total_size_in_bytes and progress_bar.n != total_size_in_bytes:
                    print("WARNING: Downloaded size does not match expected size.")

            print(f"Downloaded checkpoint to '{destination_path}'.")

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to download checkpoint: {e}")
            if destination_path.exists():
                try:
                    os.remove(destination_path)
                except OSError:
                    pass
            raise
        except IOError as e:
            print(f"ERROR: Failed to write checkpoint: {e}")
            if destination_path.exists():
                try:
                    os.remove(destination_path)
                except OSError:
                    pass
            raise

    def load_model(self):
        if not self.music_dir.exists():
            print(f"Warning: Music directory does not exist: {self.music_dir}")

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self.checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"Provided checkpoint_path '{self.checkpoint_path}' is a directory. "
                f"Expected a file path."
            )

        if not self.checkpoint_path.exists():
            self._download_checkpoint(self.checkpoint_path)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Loading CLAP model from: {self.checkpoint_path}")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=self.device)
        self.model.load_ckpt(str(self.checkpoint_path))
        print("CLAP model loaded.")

    def get_file_paths(self, extensions=(".mp3", ".wav", ".flac", ".ogg", ".m4a")):
        if not self.music_dir.exists():
            print(f"Music directory {self.music_dir} does not exist.")
            return []
        paths = [str(p.resolve()) for p in self.music_dir.rglob('*') if p.suffix.lower() in extensions]
        print(f"Found {len(paths)} audio files.")
        return paths

    def extract_metadata(self, file_path_str):
        file_path = Path(file_path_str)
        try:
            relative_path = file_path.relative_to(self.music_dir)
            parts = relative_path.parts
            if len(parts) >= 3:
                return parts[0], parts[1], file_path.stem
            elif len(parts) == 2:
                return self.music_dir.name, parts[0], file_path.stem
            elif len(parts) == 1:
                artist = self.music_dir.parent.name if self.music_dir.parent != PROJECT_ROOT else "Unknown Artist"
                return artist, self.music_dir.name, file_path.stem
            return "Unknown Artist", "Unknown Album", file_path.stem
        except ValueError:
            print(f"Warning: Could not make {file_path} relative to {self.music_dir}.")
            return "Unknown Artist", "Unknown Album", file_path.stem

    def process_files(self, file_paths_to_process):
        if not self.model:
            print("Model not loaded.")
            return []
        if not file_paths_to_process:
            print("No audio files to process.")
            return []

        num_files = len(file_paths_to_process)
        processed = []

        with torch.no_grad():
            for i in range(0, num_files, self.batch_size):
                batch = file_paths_to_process[i:i + self.batch_size]
                print(f"Processing batch {i // self.batch_size + 1}: {batch}")

                try:
                    embeddings = self.model.get_audio_embedding_from_filelist(x=batch, use_tensor=False)
                    metadata_batch = []

                    for path in batch:
                        artist, album, song = self.extract_metadata(path)
                        try:
                            stored_path = str(Path(path).relative_to(PROJECT_ROOT))
                        except ValueError:
                            stored_path = Path(path).name
                            print(f"Warning: {path} not under PROJECT_ROOT.")

                        metadata_batch.append({
                            "file_path": stored_path,
                            "artist": artist,
                            "album": album,
                            "song": song
                        })

                    for idx, data in enumerate(metadata_batch):
                        data["audio_embedding"] = embeddings[idx].tolist()
                        processed.append(data)

                    print(f"Batch processed: {len(embeddings)} embeddings.")

                except Exception as e:
                    print(f"Error processing batch starting with {batch[0]}: {e}")
                    import traceback
                    traceback.print_exc()

        self.embeddings_data.extend(processed)
        return processed

    def process_batches_from_music_dir(self):
        self.audio_file_paths = self.get_file_paths()
        if not self.audio_file_paths:
            print("No audio files found.")
            return []
        return self.process_files(self.audio_file_paths)

    def save_embeddings(self):
        if not self.embeddings_data:
            print("No embeddings to save.")
            return
        print(f"Saving {len(self.embeddings_data)} embeddings to {self.output_file}")
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.embeddings_data, f, indent=2)
            print("Embeddings saved.")
        except IOError as e:
            print(f"Error saving embeddings: {e}")
