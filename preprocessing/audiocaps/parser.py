# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

# Load a small part of the JSON to demonstrate the approach
file_path = 'data/annotations/all_caps.json'
output_path = 'data/annotations/parsed_all_caps.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    captions_data = json.load(file)

# Function to extract sources from captions
def extract_sources(caption):
    # Common patterns and their splits
    patterns = [
        " and is followed by",
        " followed by ",
        " after which",
        " which is ",
        " proceeded by ",
        " at a close distance",
        " and ",
        ", ",
        " as ",
        " while ",
        " then ",
        " with ",
        " before ",
        " after ",
        " during ",
        " nearby ",
        " over ",
        " among ",
        " precedes ",
        " proceeded ",
        " which "
    ]
    
    # Avoid splitting phrases that don't represent distinct sound sources
    avoid_split_phrases = ["on and off", "man and woman", "and is", "bells and a horn", "clanking and clunking", "and continuously"]

    sources = [caption]  # Start with the full caption as the only source

    # Attempt to split by each pattern
    for pattern in patterns:
        new_sources = []
        for source in sources:
            # Check if source contains phrases that should not lead to a split
            if pattern == " and " and any(phrase in source for phrase in avoid_split_phrases):
                new_sources.append(source)
            else:
                new_sources.extend(source.split(pattern))

        sources = new_sources

    # Filter out empty strings and strip whitespace
    sources = [source.strip() for source in sources if source.strip()]

    # Post-processing steps: Remove "then " at the beginning,
    sources = [source.replace("then ", "").replace("then", "").replace(",", "") for source in sources if source.strip()]
    
    # Post-processing steps:  capitalize the first letter, and remove any empty strings
    sources = [source.capitalize().strip() for source in sources if source.strip()]

    sources = [source.replace("And ", "").capitalize().strip() for source in sources if source.strip()]

    sources = [source.strip() for source in sources if source.strip()]

    return sources

# Process each caption to extract sources
processed_data = {}
for id, entry in captions_data.items():
    caption = entry['caption']
    file_id = entry["file_id"]
    sources = extract_sources(caption)
    processed_data[id] = {"file_id": file_id, "caption": caption, "sources": sources}

# Write to a new JSON file
with open(output_path, 'w') as outfile:
    json.dump(processed_data, outfile, indent=4)