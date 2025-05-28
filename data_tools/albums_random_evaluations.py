import requests
from bs4 import BeautifulSoup
import random
import time
import csv

def scrape_album_data(url):
    """Scrapes album data from a single page and returns a list of (album, artist, score) tuples."""
    album_data = []
    try:
        time.sleep(random.uniform(1, 3))
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        albums = soup.find_all('tr')  # Find all table rows

        for album_row in albums:
            if album_row.find('td', class_='clamp-summary-wrap'): # filter needed row
                try:
                    album_name_element = album_row.find('a', class_='title')
                    album_name = album_name_element.text.strip() if album_name_element else "Unknown Album"

                    artist_name_element = album_row.find('div', class_='artist')
                    artist_name = artist_name_element.text.strip() if artist_name_element else "Unknown Artist"

                    score_element = album_row.find('div', class_='metascore_w')
                    score = int(score_element.text.strip()) if score_element else None

                    album_data.append((album_name, artist_name, score))

                except AttributeError as e:
                    print(f"Error parsing album data: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return album_data

def scrape_multiple_pages(base_url, num_pages):
    """Scrapes a specified number of pages from a base URL."""
    all_album_data = []
    for page in range(num_pages):
        page_url = f"{base_url}?page={page}"
        album_data = scrape_album_data(page_url)
        all_album_data.extend(album_data)
        print(f"Scraped {len(album_data)} albums from {page_url}")
    return all_album_data


def filter_album_data(album_data, min_score=0, max_score=100):
    """Filters the album data to include only albums within the specified score range."""
    filtered_data = [(album, artist, score) for album, artist, score in album_data if score is not None and min_score <= score <= max_score]
    return filtered_data

def save_to_csv(album_data, filename="combined_albums.csv"):
    """Saves album data to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Album", "Artist", "Score"])
        for album, artist, score in album_data:
            writer.writerow([album, artist, score])
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    rap_url = "https://www.metacritic.com/browse/albums/genre/date/rap"
    rap_pages = 18
    pop_url = "https://www.metacritic.com/browse/albums/genre/date/pop?view=detailed"
    pop_pages = 11

    rap_data = scrape_multiple_pages(rap_url, rap_pages)
    print(f"Total number of rap albums scraped: {len(rap_data)}")
    pop_data = scrape_multiple_pages(pop_url, pop_pages)
    print(f"Total number of pop albums scraped: {len(pop_data)}")

    combined_data = rap_data + pop_data
    print(f"Total number of albums scraped (combined): {len(combined_data)}")

    min_acceptable_score = 0
    max_acceptable_score = 100
    filtered_albums = filter_album_data(combined_data, min_acceptable_score, max_acceptable_score)
    print(f"Total number of albums after filtering: {len(filtered_albums)}")

    random.shuffle(filtered_albums)
    save_to_csv(filtered_albums, "data/raw/combined_albums.csv")
    print("Scraped and saved combined albums to CSV file.")