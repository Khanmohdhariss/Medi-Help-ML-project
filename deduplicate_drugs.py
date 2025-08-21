import json

# Load the original drug names data
with open("drug_names.json", "r") as file:
    drug_data = json.load(file)

# Remove duplicates and merge dosages
deduplicated_data = {}
for drug_name, dosages in drug_data.items():
    if drug_name not in deduplicated_data:
        deduplicated_data[drug_name] = set(dosages)
    else:
        deduplicated_data[drug_name].update(dosages)

# Save the deduplicated data
with open("drug_names_deduped.json", "w") as outfile:
    json.dump(deduplicated_data, outfile)
