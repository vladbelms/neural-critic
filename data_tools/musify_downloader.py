import json
import requests
import os
import time
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import asyncio
import aiohttp
import aiofiles
from functools import lru_cache

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FastMusifyDownloader:
    def __init__(self, download_folder="data/raw/albums", max_concurrent=8):
        self.base_url = "https://musify.club"
        self.search_url = f"{self.base_url}/search"
        self.download_folder = download_folder
        self.max_concurrent = max_concurrent
        self.session_lock = Lock()

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        os.makedirs(self.download_folder, exist_ok=True)
        self.search_cache = {}

    @lru_cache(maxsize=1000)
    def sanitize_filename(self, filename):
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        return filename

    def search_track(self, artist, song_title):
        cache_key = f"{artist}|||{song_title}".lower()
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        query = f"{artist} {song_title}"
        try:
            search_params = {'searchText': query}
            with self.session_lock:
                response = self.session.get(self.search_url, params=search_params, timeout=10)
                response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            track_links = soup.find_all('a', href=lambda x: x and '/track/' in x)
            for link in track_links:
                text_content = link.get_text(separator=" ", strip=True).lower()
                if artist.lower() in text_content and song_title.lower() in text_content:
                    href = link.get('href', '')
                    track_url = self.base_url + href if not href.startswith('http') else href
                    self.search_cache[cache_key] = track_url
                    return track_url
            if track_links:
                href = track_links[0].get('href', '')
                track_url = self.base_url + href if not href.startswith('http') else href
                self.search_cache[cache_key] = track_url
                return track_url
            self.search_cache[cache_key] = None
            return None
        except Exception as e:
            print(f"Error searching for {query}: {str(e)}")
            self.search_cache[cache_key] = None
            return None

    def get_download_link(self, track_url):
        try:
            with self.session_lock:
                response = self.session.get(track_url, timeout=10)
                response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            selectors = [
                'div.playlist_actions.track_page > a.songplay_btn',
                'a.songplay_btn[href*="download"]', 'a.btn[href*="download"]',
                'a[class*="songplay_btn"]', '.playlist_actions a.btn',
                'a.btn-outline-primary'
            ]
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    text = element.get_text().lower()
                    if href and any(keyword in text for keyword in ['скачать', 'download', 'mp3']):
                        download_url = self.base_url + href if not href.startswith('http') else href
                        return download_url
            return None
        except Exception as e:
            print(f"Error getting download link from {track_url}: {str(e)}")
            return None

    def download_file_sync(self, url, file_path):
        try:
            if not url.startswith('http'): url = self.base_url + url
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)

            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                print(f"Skipping HTML page mistaken for download: {url}")
                return False
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    if chunk: f.write(chunk)
            file_size = os.path.getsize(file_path)
            if file_size < 50 * 1024:
                os.remove(file_path)
                print(f"Removed small/invalid file: {file_path} ({file_size} bytes)")
                return False
            return True
        except Exception as e:
            print(f"Error downloading file {url}: {str(e)}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            return False

    def download_single_track(self, args_tuple):
        artist, song, album_folder, track_num, total_tracks = args_tuple
        print(f"[{track_num}/{total_tracks}] Sync: Processing: {artist} - {song}")
        safe_filename = self.sanitize_filename(f"{song}.mp3")
        file_path = os.path.join(album_folder, safe_filename)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 50 * 1024:
            print(f"Sync: File already exists: {safe_filename}")
            return True, song
        track_url = self.search_track(artist, song)
        if not track_url:
            print(f"Sync: Could not find: {artist} - {song}")
            return False, song
        download_url = self.get_download_link(track_url)
        if not download_url:
            print(f"Sync: No download link: {artist} - {song}")
            return False, song
        if self.download_file_sync(download_url, file_path):
            print(f"Sync: Downloaded: {artist} - {song}")
            return True, song
        else:
            print(f"Sync: Download failed: {artist} - {song}")
            return False, song

    def download_album_tracks_concurrent(self, album_data):
        album_name = list(album_data.keys())[0]
        album_info = album_data[album_name]
        artist = album_info['artist']
        songs = album_info['songs']
        print(f"\nProcessing Album (Sync): {album_name} by {artist} ({len(songs)} tracks)")
        artist_folder = os.path.join(self.download_folder, self.sanitize_filename(artist))
        album_folder = os.path.join(artist_folder, self.sanitize_filename(album_name))
        os.makedirs(album_folder, exist_ok=True)

        download_args = [(artist, song, album_folder, i + 1, len(songs)) for i, song in enumerate(songs)]
        successful_downloads, failed_downloads = 0, 0

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_song = {executor.submit(self.download_single_track, args): args[1] for args in download_args}
            for future in as_completed(future_to_song):
                try:
                    success, _ = future.result()
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                except Exception as e:
                    print(f"Sync: Exception for {future_to_song[future]}: {str(e)}")
                    failed_downloads += 1
        print(f"\nAlbum '{album_name}' (Sync) completed! Successful: {successful_downloads}, Failed: {failed_downloads}")
        return successful_downloads, failed_downloads

    def pre_scan_existing_files(self, albums_data):
        print("Sync: Pre-scanning existing files...")
        total_tracks_overall = 0
        existing_files_count = 0
        tracks_to_download_list = []

        for album_name, album_info in albums_data.items():
            artist = album_info['artist']
            songs = album_info['songs']
            artist_folder = os.path.join(self.download_folder, self.sanitize_filename(artist))
            album_folder = os.path.join(artist_folder, self.sanitize_filename(album_name))

            current_album_tracks_to_download = []
            for song in songs:
                total_tracks_overall += 1
                safe_filename = self.sanitize_filename(f"{song}.mp3")
                file_path = os.path.join(album_folder, safe_filename)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 50 * 1024:
                    existing_files_count += 1
                else:
                    current_album_tracks_to_download.append(song)

            if current_album_tracks_to_download:
                tracks_to_download_list.append({
                    album_name: {
                        "artist": artist,
                        "songs": current_album_tracks_to_download
                    }
                })

        print(f"Sync: Found {existing_files_count}/{total_tracks_overall} files already downloaded.")
        return tracks_to_download_list, total_tracks_overall - existing_files_count


class AsyncMusifyDownloader:
    def __init__(self, download_folder="data/raw/albums", max_concurrent_downloads=10, max_concurrent_requests=10):
        self.base_url = "https://musify.club"
        self.search_url = f"{self.base_url}/search"
        self.download_folder = download_folder

        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.download_semaphore = asyncio.Semaphore(max_concurrent_downloads)

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        os.makedirs(self.download_folder, exist_ok=True)

        self.search_cache = {}
        self.search_cache_lock = asyncio.Lock()

    @lru_cache(maxsize=1000)
    def sanitize_filename(self, filename):
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        return filename

    async def _fetch_html(self, session, url, params=None, timeout_seconds=10):
        try:
            async with self.request_semaphore:
                async with session.get(url, params=params, headers=self.headers, timeout=timeout_seconds,
                                       ssl=False) as response:
                    response.raise_for_status()
                    return await response.text()
        except asyncio.TimeoutError:
            print(f"Timeout error fetching {url}")
        except aiohttp.ClientError as e:
            print(f"Client error fetching {url}: {e}")
        return None

    async def search_track_async(self, session, artist, song_title):
        cache_key = f"{artist}|||{song_title}".lower()

        async with self.search_cache_lock:
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]

        query = f"{artist} {song_title}"
        html_content = await self._fetch_html(session, self.search_url, params={'searchText': query})

        if not html_content:
            async with self.search_cache_lock:
                self.search_cache[cache_key] = None
            return None

        soup = BeautifulSoup(html_content, 'html.parser')

        track_links = soup.find_all('a', href=lambda x: x and '/track/' in x)
        found_url = None
        for link in track_links:
            text_content = link.get_text(separator=" ", strip=True).lower()
            if artist.lower() in text_content and song_title.lower() in text_content:
                href = link.get('href', '')
                found_url = self.base_url + href if not href.startswith('http') else href
                break

        if not found_url and track_links:
            href = track_links[0].get('href', '')
            found_url = self.base_url + href if not href.startswith('http') else href

        async with self.search_cache_lock:
            self.search_cache[cache_key] = found_url
        return found_url

    async def get_download_link_async(self, session, track_url):
        html_content = await self._fetch_html(session, track_url)
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')
        selectors = [
            'div.playlist_actions.track_page > a.songplay_btn',
            'a.songplay_btn[href*="download"]', 'a.btn[href*="download"]',
            'a[class*="songplay_btn"]', '.playlist_actions a.btn',
            'a.btn-outline-primary'
        ]
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                text = element.get_text().lower()
                if href and any(keyword in text for keyword in ['скачать', 'download', 'mp3']):
                    download_url = self.base_url + href if not href.startswith('http') else href
                    return download_url
        return None

    async def download_file_async(self, session, url, file_path):
        try:
            if not url.startswith('http'): url = self.base_url + url

            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)

            async with self.download_semaphore:
                timeout = aiohttp.ClientTimeout(total=120, connect=20)
                async with session.get(url, headers=self.headers, timeout=timeout, ssl=False) as response:
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '').lower()
                    if 'html' in content_type:
                        print(f"Skipping HTML page mistaken for download: {url}")
                        return False

                    async with aiofiles.open(file_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(8192 * 4)
                            if not chunk:
                                break
                            await f.write(chunk)

            file_size = os.path.getsize(file_path)
            if file_size < 50 * 1024:
                await aiofiles.os.remove(file_path)
                print(f"Removed small/invalid file: {file_path} ({file_size} bytes)")
                return False
            return True
        except asyncio.TimeoutError:
            print(f"Timeout downloading file: {url}")
        except aiohttp.ClientError as e:
            print(f"Client error downloading file {url}: {e}")
        except Exception as e:
            print(f"Generic error downloading file {url}: {str(e)}")

        try:
            if await aiofiles.os.path.exists(file_path):
                await aiofiles.os.remove(file_path)
        except Exception:
            pass
        return False

    async def download_single_track_async(self, session, artist, song, album_name, track_num, total_tracks):
        print(f"[{track_num}/{total_tracks}] Async: Processing: {artist} - {song}")

        artist_folder = os.path.join(self.download_folder, self.sanitize_filename(artist))
        album_folder = os.path.join(artist_folder, self.sanitize_filename(album_name))

        safe_filename = self.sanitize_filename(f"{song}.mp3")
        file_path = os.path.join(album_folder, safe_filename)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 50 * 1024:
            print(f"Async: File already exists: {safe_filename}")
            return True, song

        track_url = await self.search_track_async(session, artist, song)
        if not track_url:
            print(f"Async: Could not find: {artist} - {song}")
            return False, song

        download_url = await self.get_download_link_async(session, track_url)
        if not download_url:
            print(f"Async: No download link: {artist} - {song}")
            return False, song

        if await self.download_file_async(session, download_url, file_path):
            print(f"Async: Downloaded: {artist} - {song}")
            return True, song
        else:
            print(f"Async: Download failed: {artist} - {song}")
            return False, song

    async def download_album_tracks_async(self, session, album_data_item):
        album_name = list(album_data_item.keys())[0]
        album_info = album_data_item[album_name]
        artist = album_info['artist']
        songs = album_info['songs']

        print(f"\nProcessing Album (Async): {album_name} by {artist} ({len(songs)} tracks)")

        artist_folder = os.path.join(self.download_folder, self.sanitize_filename(artist))
        album_folder = os.path.join(artist_folder, self.sanitize_filename(album_name))
        os.makedirs(album_folder, exist_ok=True)

        tasks = []
        for i, song in enumerate(songs):
            task = self.download_single_track_async(
                session, artist, song, album_name, i + 1, len(songs)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_downloads = 0
        failed_downloads = 0
        for res in results:
            if isinstance(res, Exception):
                print(f"Task resulted in exception: {res}")
                failed_downloads += 1
            elif isinstance(res, tuple) and len(res) == 2:
                success, _ = res
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
            else:
                print(f"Unexpected result from task: {res}")
                failed_downloads += 1

        print(f"\nAlbum '{album_name}' (Async) completed! Successful: {successful_downloads}, Failed: {failed_downloads}")
        return successful_downloads, failed_downloads

    async def process_all_albums_async(self, albums_data_list):
        conn = aiohttp.TCPConnector(limit=100, limit_per_host=20, ssl=False)
        async with aiohttp.ClientSession(connector=conn) as session:
            total_successful_overall = 0
            total_failed_overall = 0

            for album_data_item in albums_data_list:
                successful, failed = await self.download_album_tracks_async(session, album_data_item)
                total_successful_overall += successful
                total_failed_overall += failed

            return total_successful_overall, total_failed_overall

    async def pre_scan_existing_files_async(self, albums_data):
        print("Async: Pre-scanning existing files...")
        total_tracks_overall = 0
        existing_files_count = 0
        tracks_to_download_list = []

        for album_name_key, album_info_val in albums_data.items():
            artist = album_info_val['artist']
            songs = album_info_val['songs']

            artist_folder = os.path.join(self.download_folder, self.sanitize_filename(artist))
            album_folder = os.path.join(artist_folder, self.sanitize_filename(album_name_key))

            current_album_tracks_to_download = []
            for song in songs:
                total_tracks_overall += 1
                safe_filename = self.sanitize_filename(f"{song}.mp3")
                file_path = os.path.join(album_folder, safe_filename)

                if os.path.exists(file_path) and os.path.getsize(file_path) > 50 * 1024:
                    existing_files_count += 1
                else:
                    current_album_tracks_to_download.append(song)

            if current_album_tracks_to_download:
                tracks_to_download_list.append({
                    album_name_key: {
                        "artist": artist,
                        "songs": current_album_tracks_to_download
                    }
                })

        print(f"Async: Found {existing_files_count}/{total_tracks_overall} files already downloaded.")
        return tracks_to_download_list, total_tracks_overall - existing_files_count


def load_json_data(filename="genius_albums_structured_data.json"):
    try:
        possible_paths = [
            filename,
            os.path.join("data", filename),
            os.path.join("data", "raw", filename),
            os.path.join("../data/raw", filename),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading JSON data from: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)

        print(f"Could not find '{filename}' in any location")
        return None
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None


async def async_main():
    albums_data_full = load_json_data()
    if albums_data_full is None:
        print("Failed to load album data. Exiting...")
        return

    print(f"Successfully loaded {len(albums_data_full)} albums from JSON file")

    downloader = AsyncMusifyDownloader(max_concurrent_downloads=10, max_concurrent_requests=5)

    albums_to_process_list, num_files_to_download = await downloader.pre_scan_existing_files_async(albums_data_full)

    if num_files_to_download == 0:
        print("All files already downloaded. Nothing to do!")
        return

    print(f"Need to download {num_files_to_download} files across {len(albums_to_process_list)} albums/parts of albums.")

    start_time = time.time()

    total_successful, total_failed = await downloader.process_all_albums_async(albums_to_process_list)

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nALL ASYNC DOWNLOADS COMPLETED!")
    print(f"Total successful downloads: {total_successful}")
    print(f"Total failed downloads: {total_failed}")
    print(f"Total time: {duration:.2f} seconds")
    if (total_successful + total_failed) > 0:
        print(f"Average time per file: {duration / (total_successful + total_failed):.2f} seconds")
    print(f"Files saved in: data/raw/albums/")


def sync_main():
    albums_data = load_json_data()
    if albums_data is None:
        print("Failed to load album data. Exiting...")
        return
    print(f"Successfully loaded {len(albums_data)} albums from JSON file")

    downloader = FastMusifyDownloader(max_concurrent=8)

    albums_to_process_list, num_files_to_download = downloader.pre_scan_existing_files(albums_data)

    if num_files_to_download == 0:
        print("All files already downloaded. Nothing to do!")
        return

    print(f"Need to download {num_files_to_download} files across {len(albums_to_process_list)} albums/parts of albums.")

    total_successful = 0
    total_failed = 0
    start_time = time.time()

    for album_dict in albums_to_process_list:
        successful, failed = downloader.download_album_tracks_concurrent(album_dict)
        total_successful += successful
        total_failed += failed

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nALL SYNC DOWNLOADS COMPLETED!")
    print(f"Total successful downloads: {total_successful}")
    print(f"Total failed downloads: {total_failed}")
    print(f"Total time: {duration:.2f} seconds")
    if (total_successful + total_failed) > 0:
        print(f"Average time per file: {duration / (total_successful + total_failed):.2f} seconds")
    print(f"Files saved in: data/raw/albums/")


if __name__ == "__main__":
    print("Musify.club Album Downloader")

    use_async = True

    if use_async:
        print("Using ASYNC Downloader")
        asyncio.run(async_main())
    else:
        print("Using SYNC Downloader")
        sync_main()