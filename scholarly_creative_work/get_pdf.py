import csv
from tqdm import tqdm
import os
import pandas as pd
import json
import subprocess

not_downloadable = []
Invalid_DOIs = []
downloaded_papers = []

mirrors = [
    "https://sci-hub.ee",
    "https://sci-hub.ren",
    "https://sci-hub.wf"
]


df = pd.read_csv('filtered_AC_publications.csv', encoding='ISO-8859-1', low_memory=False, delimiter=',')

for i, row in tqdm(df.iterrows(), total=len(df)):
    if i > 1600: # 1100 - 1500 
        doi_url = f"https://doi.org/{row['DOI']}"
        if not pd.isna(row['DOI']) and "(" not in doi_url and  ")" not in doi_url:
            output_path = f"papers/{row['DOI'].replace('/', '_')}.pdf"  
            if not os.path.exists(output_path):
                command = f'scidownl download --doi {row["DOI"]} --out {output_path}'
                # command = f'python -m PyPaperBot --doi {doi_url} --dwn-dir papers/'
                try:
                    print(row["DOI"])
                    subprocess.run(command, shell=True, check=True, timeout=50)
                except Exception as e:
                    print(e)
                    pass
                # !scidownl download --doi {row['DOI']} --out doi_url

                if not os.path.exists(output_path):
                    print(f"\nDownload failed for PaperID {row['ID']} with DOI: {row['DOI']}.")
                    not_downloadable.append((row['ID'], row['DOI']))
                else: 
                    downloaded_papers.append((row['ID'], row['DOI']))
            output_path = f"papers/{row['DOI'].replace('/', '_')}.pdf"
        else:
            print(f"\nPaper ID {row['ID']} doesn't have the DOI.")
            Invalid_DOIs.append(row['ID'])

    # if i == 50:
    #     break

stats = {'Downloaded': downloaded_papers,
        'Non-downloaded': not_downloadable,
        'Invalid_DOIs': Invalid_DOIs}

with open('stats.json', 'w') as f:
    json.dump(stats, f, indent=4)
