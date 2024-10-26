import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/117.0'
}

urls = [
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/people-group/anti-harassment/",
    "https://handbook.gitlab.com/handbook/people-group/givelab-volunteer-initiatives/",
    "https://handbook.gitlab.com/handbook/hiring/",
    "https://handbook.gitlab.com/handbook/company/culture/inclusion/",
    "https://handbook.gitlab.com/handbook/labor-and-employment-notices/",
    "https://handbook.gitlab.com/handbook/leadership/",
    "https://handbook.gitlab.com/handbook/people-group/learning-and-development/",
    "https://handbook.gitlab.com/handbook/people-group/general-onboarding/",
    "https://handbook.gitlab.com/handbook/people-group/offboarding/",
    "https://handbook.gitlab.com/handbook/finance/spending-company-money/",
    "https://handbook.gitlab.com/handbook/people-group/talent-assessment/",
    "https://handbook.gitlab.com/handbook/people-group/team-member-relations/#team-member-relations-philosophy",
    "https://handbook.gitlab.com/handbook/total-rewards/",
    "https://handbook.gitlab.com/handbook/tools-and-tips/"
]


def fetch_page_content(url):
    """Fetch the page content from the specified URL."""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def parse_table(table):
    """Parse an HTML table and return it as a list of dictionaries (or rows)."""
    rows = []
    headers = []

    # Find all rows in the table
    table_rows = table.find_all('tr')

    # Extract headers if they exist
    header_row = table.find('thead')
    if header_row:
        headers = [header.get_text(strip=True)
                   for header in header_row.find_all('th')]

    # Loop through all rows in the table
    for row in table_rows:
        cells = row.find_all(['th', 'td'])
        row_data = [cell.get_text(strip=True) for cell in cells]

        if headers and len(row_data) == len(headers):
            # If headers exist, create a dictionary mapping headers to row data
            row_dict = dict(zip(headers, row_data))
            rows.append(row_dict)
        else:
            # Otherwise, treat it as a list of row values
            rows.append(row_data)

    return rows


def parse_page(content, source_url):
    """Parse the webpage and extract useful data, including tables."""
    soup = BeautifulSoup(content, 'html.parser')

    content_area = soup.find('main')
    if not content_area:
        print(f"Main content area not found for URL: {source_url}")
        return None

    parsed_data = {
        'source_link': source_url,
        'content': {},
        'tables': []  # Add a list to store parsed tables
    }

    # Extract text from the content area and store it in the parsed_data dictionary
    for section in content_area.find_all(['h1', 'h2', 'h3', 'p']):
        tag_name = section.name
        text = section.get_text(strip=True)
        parsed_data['content'].setdefault(tag_name, []).append(text)

    # Find and parse all tables
    tables = content_area.find_all('table')
    for table in tables:
        table_data = parse_table(table)
        if table_data:
            parsed_data['tables'].append(table_data)

    return parsed_data


def scrape_sections():
    """Main function to scrape multiple page sections."""
    all_scraped_data = []
    for url in urls:
        print(f"Scraping URL: {url}")
        content = fetch_page_content(url)
        if content:
            parsed_data = parse_page(content, url)
            if parsed_data:
                all_scraped_data.append(parsed_data)
        else:
            print(f"Failed to retrieve page content for {url}.")
    return all_scraped_data


if __name__ == "__main__":
    scraped_data = scrape_sections()
    for data in scraped_data:
        print(f"Data from {data['source_link']}:")
        print(data['content'])
        print("Tables:")
        for table in data['tables']:
            print(table)
