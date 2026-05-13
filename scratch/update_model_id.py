
import json

with open('BioBERT_disease_mapping.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 'dmis-lab/biobert-v1.1-pubmed' in line:
                line = line.replace('dmis-lab/biobert-v1.1-pubmed', 'dmis-lab/biobert-v1.1')
            new_source.append(line)
        cell['source'] = new_source

with open('BioBERT_disease_mapping.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated BioBERT model ID to dmis-lab/biobert-v1.1")
