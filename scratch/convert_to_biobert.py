
import json
import os

# Load the original notebook
with open('TF-IDF_disease_mapping.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Title and Overview
if len(nb['cells']) > 0 and nb['cells'][0]['cell_type'] == 'markdown':
    source = nb['cells'][0]['source']
    new_source = []
    for line in source:
        line = line.replace('TF-IDF', 'BioBERT')
        new_source.append(line)
    nb['cells'][0]['source'] = new_source

# 2. Update Imports and results_dir
if len(nb['cells']) > 1 and nb['cells'][1]['cell_type'] == 'code':
    source = nb['cells'][1]['source']
    new_source = []
    added_imports = False
    for line in source:
        if 'import pandas' in line and not added_imports:
            new_source.append("import torch\n")
            new_source.append("from transformers import AutoTokenizer, AutoModel\n")
            added_imports = True
        
        line = line.replace("results_dir = 'TF-IDF Results'", "results_dir = 'BioBERT Results'")
        new_source.append(line)
    nb['cells'][1]['source'] = new_source

# 3. Replace Step 3 Markdown
if len(nb['cells']) > 6 and nb['cells'][6]['cell_type'] == 'markdown':
    nb['cells'][6]['source'] = [
        "## Step 3: BioBERT Feature Extraction\n",
        "Extracting high-dimensional embeddings using a pre-trained BioBERT model (specifically `dmis-lab/biobert-v1.1-pubmed`)."
    ]

# 4. Replace Step 3 Code
if len(nb['cells']) > 7 and nb['cells'][7]['cell_type'] == 'code':
    nb['cells'][7]['source'] = [
        "print(\"Loading BioBERT model...\")\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1-pubmed')\n",
        "model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1-pubmed').to(device)\n",
        "\n",
        "def get_embeddings(text_list, batch_size=16):\n",
        "    model.eval()\n",
        "    all_embeddings = []\n",
        "    for i in tqdm(range(0, len(text_list), batch_size)):\n",
        "        batch_texts = text_list[i:i+batch_size]\n",
        "        # Handle potential NaNs or empty strings\n",
        "        batch_texts = [str(t) if pd.notna(t) and t != '' else 'empty' for t in batch_texts]\n",
        "        \n",
        "        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)\n",
        "        with torch.no_grad():\n",
        "            outputs = model(**inputs)\n",
        "            # Mean pooling of the last hidden states\n",
        "            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()\n",
        "        all_embeddings.append(embeddings)\n",
        "    return np.vstack(all_embeddings)\n",
        "\n",
        "print(\"Extracting embeddings (this might take some time on CPU)...\")\n",
        "# Note: BERT models generally perform better on raw text rather than heavily preprocessed tokens.\n",
        "biobert_matrix = get_embeddings(df['Abstract'].tolist())\n",
        "print(f\"BioBERT Embeddings shape: {biobert_matrix.shape}\")\n",
        "\n",
        "# Save a sample of BioBERT features for inspection\n",
        "biobert_sample = pd.DataFrame(biobert_matrix[:100])\n",
        "biobert_sample.to_csv(os.path.join(results_dir, 'step3_biobert_sample.csv'), index=False)\n",
        "print(\"BioBERT sample saved.\")"
    ]

# 5. Update Step 4 Markdown
if len(nb['cells']) > 8 and nb['cells'][8]['cell_type'] == 'markdown':
    source = nb['cells'][8]['source']
    new_source = []
    for line in source:
        line = line.replace('TF-IDF', 'BioBERT')
        new_source.append(line)
    nb['cells'][8]['source'] = new_source

# 6. Update Step 4 Code
if len(nb['cells']) > 9 and nb['cells'][9]['cell_type'] == 'code':
    source = nb['cells'][9]['source']
    new_source = []
    for line in source:
        line = line.replace('tfidf_matrix', 'biobert_matrix')
        new_source.append(line)
    nb['cells'][9]['source'] = new_source

# 7. Add Column mapping after Step 5 Clustering to ensure compatibility
if len(nb['cells']) > 11 and nb['cells'][11]['cell_type'] == 'code':
    source = nb['cells'][11]['source']
    source.append("\n# Map main results to a generic 'Cluster' column for downstream steps\n")
    source.append("df['Cluster'] = df['KMeans_Cluster']\n")
    source.append("n_clusters = optimal_k\n")
    nb['cells'][11]['source'] = source

# Clear all outputs to keep it fresh
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# Save the new notebook
with open('BioBERT_disease_mapping.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("BioBERT_disease_mapping.ipynb created successfully.")
