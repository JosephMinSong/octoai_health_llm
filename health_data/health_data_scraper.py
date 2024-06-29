import requests
from bs4 import BeautifulSoup

URL = "https://www.who.int/news-room/fact-sheets/detail/mental-disorders"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
results = soup.find(id="PageContent_T0643CD2A003_Col00")
print(results.prettify())
