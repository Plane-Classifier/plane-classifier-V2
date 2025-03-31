import requests
from bs4 import BeautifulSoup
import re
import os
import time
import json
import hashlib
from collections import defaultdict
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

BASE_URL = "https://www.airliners.net/search?photoCategory=23&sortBy=dateAccepted&sortOrder=desc&perPage=84&display=detail&page={}"
SAVE_DIR = "downloaded_planes"
STATE_FILE = "scraper_state.json"
LIMIT_PER_CLASS = 1000

AIRCRAFT_CLASSES = {
    "A300": "A310",
    "A310": "A310",
    "A318": "A318",
    "A319": "A319",
    "A320": "A320",
    "A321": "A321",
    "A330": "A330",
    "A340": "A340",
    "A350": "A350",
    "A380": "A380",
    "707": "B707",
    "727": "B727",
    "737-2": "B737 Classic",
    "737-3": "B737 Classic",
    "737-4": "B737 Classic",
    "737-5": "B737 Classic",
    "737-6": "B737 NG",
    "737-7": "B737 NG",
    "737-8": "B737 NG",
    "737-9": "B737 NG",
    "737 MAX": "B737 MAX",
    "747": "B747",
    "757": "B757",
    "767": "B767",
    "777": "B777",
    "787": "B787"
}

os.makedirs(SAVE_DIR, exist_ok=True)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            state["downloaded"] = set(state["downloaded"])
            return state
    return {"page": 1, "downloaded": set(), "progress": {}, "elapsed": 0.0}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump({
            "page": state["page"],
            "downloaded": list(state["downloaded"]),
            "progress": state["progress"],
            "elapsed": state["elapsed"]
        }, f)

def map_aircraft_type(text):
    for key, label in AIRCRAFT_CLASSES.items():
        if key in text:
            return label
    return None

def print_progress(progress):
    print("\nClass Progress:")
    for cls, count in progress.items():
        bar = tqdm(total=LIMIT_PER_CLASS, initial=count, desc=cls, ncols=80)
        bar.close()

def get_entry_data(card):
    link_tag = card.find("a", href=True)
    if not link_tag:
        return None, None
    entry_url = "https://www.airliners.net" + link_tag["href"]
    try:
        res = requests.get(entry_url, headers=HEADERS)
        if res.status_code != 200:
            return None, None
        soup = BeautifulSoup(res.content, 'html.parser')

        plane_type_div = soup.find('div', class_='pib-section-content-left')
        plane_type = None
        if plane_type_div:
            all_links = plane_type_div.find_all('a', href=True)
            if all_links:
                plane_type = all_links[-1].text.strip()

        image_tag = soup.find('div', class_='pdp-image-wrapper')
        img_url = None
        if image_tag:
            img_tag = image_tag.find('img')
            if img_tag and 'src' in img_tag.attrs:
                img_url = img_tag['src']

        return plane_type, img_url
    except:
        return None, None

def scrape_page(page_num, downloaded, progress):
    print(f"Scraping page {page_num}...")
    url = BASE_URL.format(page_num)
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        print(f"Failed to fetch page {page_num}")
        return downloaded, progress

    soup = BeautifulSoup(res.text, 'html.parser')
    cards = soup.find_all("div", class_="resultPreview")

    for card in cards:
        plane_type, img_url = get_entry_data(card)
        if not plane_type or not img_url:
            continue

        aircraft_class = map_aircraft_type(plane_type)
        if aircraft_class:
            if progress.get(aircraft_class, 0) >= LIMIT_PER_CLASS:
                continue
            if img_url in downloaded:
                continue

            hash_str = hashlib.md5(img_url.encode()).hexdigest()[:8]
            filename = os.path.join(SAVE_DIR, f"{aircraft_class}_{hash_str}.jpg")

            try:
                img_data = requests.get(img_url, headers=HEADERS).content
                with open(filename, 'wb') as f:
                    f.write(img_data)
                downloaded.add(img_url)
                progress[aircraft_class] = progress.get(aircraft_class, 0) + 1
                print_progress(progress)
            except Exception as e:
                print(f"Error downloading image: {e}")

    return downloaded, progress

def format_time(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"

if __name__ == "__main__":
    state = load_state()
    page = state["page"]
    downloaded = state["downloaded"]
    progress = defaultdict(int, state.get("progress", {}))
    total_elapsed = state.get("elapsed", 0.0)

    start_time = time.time()

    while True:
        if all(count >= LIMIT_PER_CLASS for count in progress.values() if count > 0):
            print("All classes have reached their image limit.")
            break

        page_start_time = time.time()
        downloaded, progress = scrape_page(page, downloaded, progress)
        page += 1
        total_elapsed += time.time() - page_start_time
        save_state({"page": page, "downloaded": downloaded, "progress": dict(progress), "elapsed": total_elapsed})

        print("\nEstimated time elapsed:", format_time(total_elapsed))
        time.sleep(1)