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

BASE_URL_TEMPLATE = "https://www.airliners.net/search?keywords={}&photoCategory=23&page={}"
SAVE_DIR = "downloaded_planes"
STATE_FILE = "scraper_state.json"
LIMIT_PER_CLASS = 500

AIRCRAFT_CLASSES = {
    "A300": "A300",
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
    "737-8 MAX": "B737 MAX",
    "737-9 MAX": "B737 MAX",
    "747": "B747",
    "757": "B757",
    "767": "B767",
    "777": "B777",
    "787": "B787"
}

SUBCLASS_LIMITS = {
    "B737 Classic": ["737-2", "737-3", "737-4", "737-5"],
    "B737 NG": ["737-6", "737-7", "737-8", "737-9"],
    "B737 MAX": ["737-8 MAX", "737-9 MAX"]
}

SUBCLASS_TARGETS = {
    group: LIMIT_PER_CLASS // len(subs)
    for group, subs in SUBCLASS_LIMITS.items()
}

os.makedirs(SAVE_DIR, exist_ok=True)

QUERIES = [
    "Airbus+A300", "Airbus+A310", "Airbus+A318", "Airbus+A319", "Airbus+A320",
    "Airbus+A321", "Airbus+A330", "Airbus+A340", "Airbus+A350", "Airbus+A380",
    "Boeing+707", "Boeing+727", "Boeing+737", "Boeing+737-8+MAX", "Boeing+737-9+MAX",
    "Boeing+747", "Boeing+757", "Boeing+767", "Boeing+777", "Boeing+787"
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            state["downloaded"] = set(state["downloaded"])
            return state
    return {"query_index": 0, "page": 1, "downloaded": set(), "progress": {}, "subclass_counts": {}, "elapsed": 0.0}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump({
            "query_index": state["query_index"],
            "page": state["page"],
            "downloaded": list(state["downloaded"]),
            "progress": state["progress"],
            "subclass_counts": state["subclass_counts"],
            "elapsed": state["elapsed"]
        }, f)

def map_aircraft_type(text):
    if "737-8 MAX" in text or "737-9 MAX" in text:
        return "B737 MAX", text
    elif "737-9" in text and "MAX" not in text:
        return "B737 NG", text
    for key, label in AIRCRAFT_CLASSES.items():
        if key in text:
            return label, text
    return None, None

def print_progress(progress, subclass_counts):
    print("\nClass Progress:")
    for cls, count in progress.items():
        bar = tqdm(total=LIMIT_PER_CLASS, initial=count, desc=cls, ncols=80)
        bar.close()

    print("\nSubclass Progress:")
    for key, count in subclass_counts.items():
        bar = tqdm(total=SUBCLASS_TARGETS[key.split(":")[0]], initial=count, desc=key, ncols=80)
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

def scrape_page(query, page_num, downloaded, progress, subclass_counts):
    start_time = time.time()
    MAX_TIME_PER_PAGE = 60
    MAX_RETRIES = 3

    print(f"Scraping page {page_num} for query '{query}'...")
    url = BASE_URL_TEMPLATE.format(query, page_num)

    for _ in range(MAX_RETRIES):
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                break
        except Exception as e:
            print(f"Retrying due to error: {e}")
            time.sleep(1)
    else:
        print(f"Failed to fetch page {page_num} after retries")
        return downloaded, progress, subclass_counts

    soup = BeautifulSoup(res.text, 'html.parser')
    cards = soup.find_all("div", class_="resultPreview")

    for card in cards:
        plane_type, img_url = get_entry_data(card)
        if not plane_type or not img_url:
            continue

        aircraft_class, raw_type = map_aircraft_type(plane_type)
        if not aircraft_class:
            continue

        for subclass_group, subclass_list in SUBCLASS_LIMITS.items():
            for subclass in subclass_list:
                if subclass in raw_type:
                    key = f"{subclass_group}:{subclass}"
                    if subclass_counts.get(key, 0) >= SUBCLASS_TARGETS[subclass_group]:
                        break
                    if img_url in downloaded:
                        break
                    hash_str = hashlib.md5(img_url.encode()).hexdigest()[:8]
                    folder = os.path.join(SAVE_DIR, aircraft_class, subclass)
                    os.makedirs(folder, exist_ok=True)
                    filename = os.path.join(folder, f"{aircraft_class}_{hash_str}.jpg")
                    try:
                        img_data = requests.get(img_url, headers=HEADERS).content
                        with open(filename, 'wb') as f:
                            f.write(img_data)
                        downloaded.add(img_url)
                        subclass_counts[key] = subclass_counts.get(key, 0) + 1
                        progress[aircraft_class] = progress.get(aircraft_class, 0) + 1
                        print_progress(progress, subclass_counts)
                    except Exception as e:
                        print(f"Error downloading image: {e}")
                    break
        else:
            if progress.get(aircraft_class, 0) >= LIMIT_PER_CLASS:
                continue
            if img_url in downloaded:
                continue
            hash_str = hashlib.md5(img_url.encode()).hexdigest()[:8]
            folder = os.path.join(SAVE_DIR, aircraft_class)
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"{aircraft_class}_{hash_str}.jpg")
            try:
                img_data = requests.get(img_url, headers=HEADERS).content
                with open(filename, 'wb') as f:
                    f.write(img_data)
                downloaded.add(img_url)
                progress[aircraft_class] = progress.get(aircraft_class, 0) + 1
                print_progress(progress, subclass_counts)
            except Exception as e:
                print(f"Error downloading image: {e}")

    if time.time() - start_time > MAX_TIME_PER_PAGE:
        print(f"Page {page_num} for query '{query}' took too long. Skipping early.")

    return downloaded, progress, subclass_counts

def format_time(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"

if __name__ == "__main__":
    state = load_state()
    query_index = state.get("query_index", 0)
    page = state.get("page", 1)
    downloaded = state["downloaded"]
    progress = defaultdict(int, state.get("progress", {}))
    subclass_counts = defaultdict(int, state.get("subclass_counts", {}))
    total_elapsed = state.get("elapsed", 0.0)

    while query_index < len(QUERIES):
        start_time = time.time()

        if all(count >= LIMIT_PER_CLASS for count in progress.values() if count > 0):
            print("All classes have reached their image limit.")
            break

        downloaded, progress, subclass_counts = scrape_page(QUERIES[query_index], page, downloaded, progress, subclass_counts)
        page += 1
        total_elapsed += time.time() - start_time
        save_state({
            "query_index": query_index,
            "page": page,
            "downloaded": downloaded,
            "progress": dict(progress),
            "subclass_counts": dict(subclass_counts),
            "elapsed": total_elapsed
        })

        print("\nEstimated time elapsed:", format_time(total_elapsed))
        time.sleep(1)

        if page > 100:
            query_index += 1
            page = 1