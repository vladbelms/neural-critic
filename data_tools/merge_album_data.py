import json
from pathlib import Path
import argparse
import re  # For string cleaning


def clean_song_title(title: str) -> str:
    """
    Cleans a song title for better matching.
    - Converts to lowercase
    - Removes leading/trailing whitespace
    - Removes common "Ft." or "(feat." patterns and content within parentheses/brackets
      that often denote featured artists or versions not present in the base title.
    """
    if not isinstance(title, str):
        return ""

    cleaned_title = title.lower().strip()

    cleaned_title = re.sub(r'\s*\([^)]*\)', '', cleaned_title).strip()
    cleaned_title = re.sub(r'\s*\[[^\]]*\]', '', cleaned_title).strip()


    return cleaned_title


def merge_data(clap_embeddings_path: Path, albums_structured_path: Path, output_path: Path):
    """
    Merges song embeddings with album structural data.

    Args:
        clap_embeddings_path (Path): Path to JSON file with song details and embeddings.
                                     (list of song objects)
        albums_structured_path (Path): Path to JSON file with album structure.
                                       (dictionary with album titles as keys)
        output_path (Path): Path to save the merged JSON output.
    """
    print(f"Loading song embeddings from: {clap_embeddings_path}")
    try:
        with open(clap_embeddings_path, 'r', encoding='utf-8') as f:
            song_details_list = json.load(f)  # This is a list of song objects
    except FileNotFoundError:
        print(f"Error: Clap embeddings file not found at {clap_embeddings_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {clap_embeddings_path}")
        return

    print(f"Loading album structures from: {albums_structured_path}")
    try:
        with open(albums_structured_path, 'r', encoding='utf-8') as f:
            albums_info_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Albums structured data file not found at {albums_structured_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {albums_structured_path}")
        return

    song_lookup = {}
    print("Building song lookup from clap_music_embeddings.json...")
    for song_detail in song_details_list:
        artist_name = song_detail.get("artist", "")
        song_title = song_detail.get("song", "")

        if not artist_name or not song_title:
            print(
                f"Warning: Skipping song entry in clap_embeddings due to missing artist/song title: {song_detail.get('file_path', 'N/A')}")
            continue

        artist_key = artist_name.strip().lower()
        song_key = clean_song_title(song_title)

        if (artist_key, song_key) in song_lookup:
            pass
        song_lookup[(artist_key, song_key)] = song_detail
    print(f"Built song lookup with {len(song_lookup)} unique (artist, song_title) entries.")

    output_albums_list = []
    albums_processed_count = 0
    songs_matched_count = 0
    songs_not_found_count = 0

    print("\nProcessing albums from albums_structured_data.json...")
    for album_title_from_key, album_data_entry in albums_info_dict.items():
        album_artist_original = album_data_entry.get("artist")
        score_str = album_data_entry.get("score")
        song_titles_from_album_file = album_data_entry.get("songs", [])

        if not album_artist_original:
            print(f"Warning: Skipping album '{album_title_from_key}' due to missing artist field.")
            continue

        album_score_numeric = None
        if score_str is not None:
            try:
                album_score_numeric = float(score_str)
            except ValueError:
                print(
                    f"Warning: Could not convert score '{score_str}' to number for album '{album_title_from_key}'. Leaving as None.")

        album_artist_lower = album_artist_original.strip().lower()
        songs_with_embeddings_for_this_album = []

        for song_title_original_from_album_file in song_titles_from_album_file:
            if not isinstance(song_title_original_from_album_file, str):
                print(
                    f"Warning: Skipping non-string song title '{song_title_original_from_album_file}' in album '{album_title_from_key}' by '{album_artist_original}'")
                continue

            cleaned_song_title_for_lookup = clean_song_title(song_title_original_from_album_file)
            lookup_key = (album_artist_lower, cleaned_song_title_for_lookup)
            matched_song_detail_from_clap = song_lookup.get(lookup_key)

            if matched_song_detail_from_clap:
                songs_matched_count += 1
                album_title_from_clap = matched_song_detail_from_clap.get("album", "")

                song_data_for_output = {
                    "song_title": matched_song_detail_from_clap.get("song", song_title_original_from_album_file),
                    "audio_embedding": matched_song_detail_from_clap.get("audio_embedding"),
                    "text_embedding_prompt": matched_song_detail_from_clap.get("text_embedding_prompt"),
                    "text_embedding": matched_song_detail_from_clap.get("text_embedding"),
                    "file_path": matched_song_detail_from_clap.get("file_path"),
                    "clap_album_title_info": album_title_from_clap
                }
                songs_with_embeddings_for_this_album.append(song_data_for_output)
            else:
                songs_not_found_count += 1

        if songs_with_embeddings_for_this_album:
            output_album_entry = {
                "album_title": album_title_from_key,
                "artist": album_artist_original,
                "score": album_score_numeric,
                "songs": songs_with_embeddings_for_this_album,
                "genius_url_attempted": album_data_entry.get("genius_url_attempted")
            }
            output_albums_list.append(output_album_entry)
            albums_processed_count += 1
        else:
            print(
                f"Info: No songs with embeddings found for album '{album_title_from_key}' by '{album_artist_original}' (songs listed: {len(song_titles_from_album_file)}). Skipping this album in output.")

    final_output_data = {"Albums": output_albums_list}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Summary ---")
    print(f"Total album entries in albums_structured_data: {len(albums_info_dict)}")
    print(f"Albums added to output (with at least one matched song): {albums_processed_count}")
    print(f"Total songs from album files matched with embeddings: {songs_matched_count}")
    print(f"Total songs listed in albums but not found in embeddings: {songs_not_found_count}")

    print(f"Saving merged data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(final_output_data, outfile, indent=2)
    print("Done.")


def main():

    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent

    default_clap_embeddings_path = project_root / "data" / "raw" / "clap_music_embeddings.json"
    default_albums_structured_path = project_root / "data" / "raw" / "albums_structured_data.json"
    default_output_path = project_root / "data" / "processed" / "dp.json"

    parser = argparse.ArgumentParser(description="Merge song embeddings with album structural data.")
    parser.add_argument(
        "--clap_embeddings", type=Path, default=default_clap_embeddings_path,
        help="Path to JSON file with song details and embeddings (list of song objects)."
    )
    parser.add_argument(
        "--album_structures", type=Path, default=default_albums_structured_path,
        help="Path to JSON file with album structure (dict with album titles as keys)."
    )
    parser.add_argument(
        "--output", type=Path, default=default_output_path,
        help="Path to save the merged JSON output."
    )
    args = parser.parse_args()

    merge_data(args.clap_embeddings, args.album_structures, args.output)


if __name__ == "__main__":
    main()