"""
Scrapes for radio images from encyclopaedia.org
"""

from typing import Iterable
import requests
from bs4 import BeautifulSoup
from time import sleep
from pathlib import Path
from rich import print
from rich.live import Live
from rich.table import Table
import subprocess
import json
from slugify import slugify
from helium import start_firefox
import helium

results_table = Table("title", "image", "broken certainty (%)")

def total_pages() -> int:
    """
    Return index of last page
    """
    page = BeautifulSoup(requests.get("https://radiopaedia.org/encyclopaedia/cases/trauma?lang=us&modality=X-ray").text, features="lxml")
    navigation_links = page.select_one("[role=navigation]").select("a")
    points_to_page = lambda link: int(link["href"].split("&page=")[1])


    return max(map(points_to_page, navigation_links))



def from_page(n: int) -> Iterable[tuple[str, str, float]]:
    """
    Return list of (title, image URL, broken with certainty (0.0 to 1.0))
    """
    page = BeautifulSoup(requests.get(f"https://radiopaedia.org/encyclopaedia/cases/trauma?lang=us&modality=X-ray&page={n}").text, features="lxml")

    image_id = lambda image_url: image_url.split("/")[4]
    fullsize_image_url = lambda image_id: f"https://radiopaedia.org/images/{image_id}/download?lang=us"

    for case in page.select("a.search-result-case"):

        title = case.select_one("h4").text
        image_url = fullsize_image_url(image_id(case.select_one("img.media-object")["src"]))
        certainty = {
            "Diagnosis certain": 1.0,
            "Diagnosis almost certain": 0.8,
            "Diagnosis probable": 0.6,
        }.get(case.select_one(".diagnostic-certainty-title").text.strip(), 0.0)

        results_table.add_row(title, image_url, f"{round(certainty*100)}%")

        yield title, image_url, certainty




def scrape() -> list[tuple[str, bool]]:
    results = []
    with Live(results_table):
        for n in range(1, total_pages()+1):
            try:
                results += from_page(n)
            except Exception as e:
                (Path(__file__).parent / "results/encyclopaedia.json").write_text(json.dumps([
                    {
                        "image": {
                            "url": case[1],
                            "id": int(case[1].split("/")[4]),
                        },
                        "title": case[0],
                        "certainty": case[2],
                    } for case in results
                ]))
                print("[green]early save written[/]")
                raise e
            results.sort(key=lambda case: case[0])
    
    return results


if __name__ == "__main__":
    for case in json.loads((Path(__file__).parent  / "results" / "encyclopaedia.json").read_text()):
        print(f"- [cyan]Downloading {case['image']['url']}")
        start_firefox(case["image"]["url"])
        # TODO: add in sources.yaml, cite using citation from download page
        helium.click("Accept all & visit the site")
        helium.click("Accept")
        helium.scroll_down(1_000)
        helium.click("Download")
        sleep(2)
        # todo get coordinates of save as radiobutton
        subprocess.run(["xdotool", "mousemove", "1246", "586"])
        subprocess.run(["xdotool", "click", "1"])
        subprocess.run(["xdotool", "mousemove", "1603", "685"])
        subprocess.run(["xdotool", "click", "1"])
        sleep(1)
        helium.kill_browser()
        subprocess.run(f"echo '{json.dumps(case)}' >> {Path(__file__).parent / 'results' / 'encyclopaedia-downloaded.jsonl'}", shell=True)
