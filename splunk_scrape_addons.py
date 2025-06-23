from bs4 import BeautifulSoup
import requests


url = "https://docs.splunk.com/Documentation/AddOns"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

response = requests.get(url, headers=headers)
if response.status_code != 200:
    print(f"Failed to retrieve page. Status code: {response.status_code}")
    exit(1)

soup = BeautifulSoup(response.content, "html.parser")

# Find all list items that contain the add-ons
add_ons_dict = {}
for p in soup.find_all("p"):
    a_tag = p.find("a", href=True)
    if not a_tag:
        continue
    name = a_tag.get_text(strip=True)
    if a_tag and "/Documentation/AddOns" in a_tag['href'] and name.startswith("Splunk Add-on for "):
        link = f"https://docs.splunk.com{a_tag['href']}"
        add_ons_dict[name] = link

# Print the results
for name, link in add_ons_dict.items():
    print(f"{name}: {link}")
