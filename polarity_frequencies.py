# Some parts are taken from: https://github.com/QuintenvdVijver/Ontology-Augmented-Prompt-Engineering/blob/main/Data/polarity%20frequencies.py
import xml.etree.ElementTree as ET
import json
from collections import Counter

def polarity_frequencies(input_path):
    polarity_counts = Counter()
    
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Go through all Term elements
    for opinion in root.findall('.//Opinion'):
        polarity = opinion.get('polarity')
        if polarity:
            polarity_counts[polarity.lower()] += 1

    # Calculate total for frequency percentages
    total = sum(polarity_counts.values())

    polarity_frequencies = {
        polarity: count / total * 100 
        for polarity, count in polarity_counts.items()
    }

    return {
        'frequencies': {k: f"{v:.2f}%" for k, v in polarity_frequencies.items()},
        'total': total
    }

# Load for each dataset (after all the pre-processing steps)
converted_output_path = "DATA_PATH"

# Print polarity frequencies
result = polarity_frequencies(converted_output_path)
print(result)

