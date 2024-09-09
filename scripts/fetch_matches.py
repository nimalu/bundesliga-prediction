import requests
import json

YEAR_FROM = 2010
YEAR_TO = 2024

matches = []

for year in range(YEAR_FROM, YEAR_TO + 1):
    print(f"Fetching matches of season {year}")
    response = requests.get(f"https://api.openligadb.de/getmatchdata/bl1/{year}")
    matches += response.json()

with open("data/matches.json", "w") as file:
    file.write(json.dumps(matches))


print(f"Scraped {len(matches)} bl1 matches")