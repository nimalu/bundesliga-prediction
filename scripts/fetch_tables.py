import requests
import json

YEAR_FROM = 2010
YEAR_TO = 2023

bl1_tables = {}
for year in range(YEAR_FROM, YEAR_TO + 1):
    print(f"Fetching table of season {year}")
    response = requests.get(f"https://api.openligadb.de/getbltable/bl1/{year}")
    bl1_tables[str(year)] = response.json()


with open("data/tables.json", "w") as file:
    file.write(json.dumps(bl1_tables))