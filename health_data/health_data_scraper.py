import requests
from bs4 import BeautifulSoup

URL = "https://www.who.int/news-room/fact-sheets/detail/mental-disorders"
page = requests.get(URL)

print(page.text)
soup = BeautifulSoup(page.content, "html.parser")
