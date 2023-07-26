import requests
from bs4 import BeautifulSoup


class Scraper:
    def __init__(self, config):
        self.config = config

    def scrape_wiktionary(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        texts = []
        for row in soup.select('#mw-content-text > div.mw-parser-output > div.jsAdd > table > tbody')[0].find_all('tr'):
            for lists in row.find_all('ul'):
                for e in lists.find_all('li'):
                    text = e.text
                    if text != '':
                        texts.append(text)
        return texts


