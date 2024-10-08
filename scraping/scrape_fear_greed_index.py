import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

def fetch_fear_and_greed_index():
    url = 'https://alternative.me/crypto/fear-and-greed-index/'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        selector = '#main > section > div > div.columns > div:nth-child(2) > div > div > div:nth-child(1) > div:nth-child(2) > div'
        element = soup.select_one(selector)
        if element:
            return element.text.strip()
    return None

def save_to_csv(score):
    filename = "data/fear_and_greed_score.csv"
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([now, score])

if __name__ == "__main__":
    score = fetch_fear_and_greed_index()
    if score is not None:
        save_to_csv(score)
