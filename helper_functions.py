import json


def save_json(data, filename):
    """Save data as a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")
