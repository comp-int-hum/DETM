import gzip, json, math, requests
from typing import Dict, Any, Generator
import numpy as np

def open_jsonl_file(file_path: str) -> Generator[Dict[str, Any], None, None]:
    with gzip.open(file_path, mode="rt") as f:
        for line in f:
            yield line

def write_jsonl_file(dump_file_path: str, item_data: Dict[str, str]) -> None:
    with gzip.open(dump_file_path, "at", encoding="utf-8") as file:
        json_line = json.dumps(item_data)
        file.write(json_line + "\n")

def get_window_ranges(min_time, max_time, window_size, get_str=True):
    start_finish_per_window = [(min_time + idx * window_size, 
                                (min_time + (idx + 1) * window_size
                                 if (min_time + (idx + 1) * window_size <= max_time) 
                                 else max_time)
                                 )
                                for idx in range(math.ceil((max_time - min_time) / window_size))]
    if get_str:
        return [f"{start_year}-{end_year}" 
                for (start_year, end_year) in start_finish_per_window]
    else:
        return [np.arange(start_year, end_year)
                for (start_year, end_year) in start_finish_per_window]


def translate_text_deepl(text, target_lang='EN'):
    api_key = '0e1ffe54-80c7-462d-9bac-b2962074ba50:fx'
    url = 'https://api-free.deepl.com/v2/translate'
    
    data = {
        'auth_key': api_key,
        'text': text,
        'target_lang': target_lang
    }

    response = requests.post(url, data=data)
    if response.status_code == 200:
        result = response.json()
        translation = result['translations'][0]['text']
        return f"{text}({translation})"
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return text