from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from langchain.schema import Document


def load_urls(urls):
    docs = []

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        for url in urls:

            page.goto(url, wait_until="networkidle")

            html = page.content()

            soup = BeautifulSoup(html, "html.parser")

            text = soup.get_text(separator="\n", strip=True)

            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": url}
                )
            )

        browser.close()

    return docs