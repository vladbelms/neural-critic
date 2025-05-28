import csv
import requests
from bs4 import BeautifulSoup
import re
import time
import json


def slugify(text):
    """Convert a string to a URL-friendly slug."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = text.lower()
    text = text.replace("'", "").replace('"', '')
    text = re.sub(r'[^\w\s-]', '-', text)
    text = re.sub(r'[-\s]+', '-', text)
    text = text.strip('-')
    return text


def get_songs_for_album(artist_name_from_csv, album_name_from_csv):
    """Fetches song titles for a given album from Genius.com."""
    if artist_name_from_csv.lower().startswith("by "):
        artist_name_for_slug = artist_name_from_csv[3:].strip()
    else:
        artist_name_for_slug = artist_name_from_csv.strip()

    slug_artist = slugify(artist_name_for_slug)
    slug_album = slugify(album_name_from_csv)

    if not slug_artist or not slug_album:
        print(f"Skipping due to empty slug for Artist: '{artist_name_from_csv}', Album: '{album_name_from_csv}'")
        return f"INVALID_SLUG_FOR_{slug_artist}_OR_{slug_album}", []

    genius_url = f"https://genius.com/albums/{slug_artist}/{slug_album}"
    print(f"Attempting to fetch: {genius_url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    songs = []
    final_url_attempted = genius_url

    try:
        response = requests.get(genius_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {genius_url}: {e}")
        slug_album_variant = slug_album
        modified_variant = False
        if "-ep" in slug_album_variant:
            slug_album_variant = slug_album_variant.replace("-ep", "").strip('-')
            modified_variant = True
        if "-mixtape" in slug_album_variant:
            slug_album_variant = slug_album_variant.replace("-mixtape", "").strip('-')
            modified_variant = True

        if modified_variant and slug_album_variant and slug_album_variant != slug_album:
            genius_url_variant = f"https://genius.com/albums/{slug_artist}/{slug_album_variant}"
            print(f"Retrying with variant: {genius_url_variant}")
            final_url_attempted = genius_url_variant
            try:
                response = requests.get(genius_url_variant, headers=headers, timeout=15)
                response.raise_for_status()
            except requests.exceptions.RequestException as e_variant:
                print(f"Error fetching variant {genius_url_variant}: {e_variant}")
                return final_url_attempted, []
        else:
            return final_url_attempted, []

    soup = BeautifulSoup(response.content, 'html.parser')

    song_elements = soup.find_all('h3', class_='chart_row-content-title')

    if not song_elements:
        song_elements = soup.select('div[id^="defer-section-"] h3.chart_row-content-title')
        if not song_elements:
            song_elements = soup.select('div.chart_row--light_border h3')

    for song_element in song_elements:
        title_text = song_element.get_text(strip=True)
        if title_text.lower().endswith(" lyrics"):
            title_text = title_text[:-7].strip()
        title_text = re.sub(r'\s*\([\w\s]+?\)\s*$', '', title_text).strip()
        if title_text:
            songs.append(title_text)

    if not songs:
        print(f"No songs found on page for {album_name_from_csv} by {artist_name_for_slug} at {final_url_attempted}")

    return final_url_attempted, songs


csv_file_path = 'data/raw/combined_albums.csv'
all_albums_data = {}

try:
    with open(csv_file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for i, row in enumerate(reader):
            if len(row) < 3:
                print(f"Skipping malformed row: {row}")
                continue

            album_name_csv, artist_name_csv, score_csv = row[0].strip(), row[1].strip(), row[2].strip()

            cleaned_artist_for_display = artist_name_csv
            if artist_name_csv.lower().startswith("by "):
                cleaned_artist_for_display = artist_name_csv[3:].strip()

            print(f"\nProcessing ({i + 1}): Album='{album_name_csv}', Artist (original CSV)='{artist_name_csv}'")

            fetched_url, song_list = get_songs_for_album(artist_name_csv, album_name_csv)

            all_albums_data[album_name_csv] = {
                "artist": cleaned_artist_for_display,
                "songs": song_list,
                "score": score_csv,
                "genius_url_attempted": fetched_url
            }

            time.sleep(1.2)

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please make sure 'combined_albums.csv' is in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")

print("\n\n--- Collected Album Data ---")
if all_albums_data:
    for album_title_key, album_details in all_albums_data.items():
        print(f"\nAlbum: {album_title_key}")
        print(f"  Artist: {album_details['artist']}")
        print(f"  Score: {album_details['score']}")
        print(f"  Genius URL Attempted: {album_details['genius_url_attempted']}")
        if album_details['songs']:
            print("  Songs Found:")
            for idx, song in enumerate(album_details['songs']):
                print(f"    {idx + 1}. {song}")
        else:
            print("  No songs found for this album on Genius.")
else:
    print("No data was processed or collected.")

json_output_file = '../data/raw/genius_albums_structured_data.json'
try:
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_albums_data, f, ensure_ascii=False, indent=4)
    print(f"\nAll album data also saved to: {json_output_file}")
except Exception as e:
    print(f"\nError saving data to JSON file: {e}")