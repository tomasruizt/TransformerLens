import json
import os
import requests
import pandas as pd

url = "https://www.neuronpedia.org/api/explanation/export"

headers = {"X-Api-Key": os.environ.get("NEURONPEDIA_API_KEY")}
params = {"saeId": "20-gemmascope-res-16k", "modelId": "gemma-2-2b"}

response = requests.get(url, headers=headers, params=params)
df = pd.DataFrame(response.json())
df.to_parquet("features/explanations.parquet")
with open("features/params.json", "w") as f:
    json.dump(params, f)
