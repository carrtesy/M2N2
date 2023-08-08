import os
import json
import numpy as np
import scipy.stats as st
import pandas as pd
import sys

path = sys.argv[1]

conf = 0.95
exps = [d for d in os.listdir(path) if "SEED" in d]
print(f"exps: {exps}")

final_results = {
    "samples": len(exps),
    "confidence": conf,
    "results": {},
}

for d in exps:
    with open(os.path.join(path, d, "result.json"), 'r') as f:
        result = json.load(f)

    for k in result:
        if k in final_results["results"]:
            final_results["results"][k].append(result[k])
        else:
            final_results["results"][k] = [result[k]]

with open(os.path.join(path, "combined_results.json"), "w") as f:
    json.dump(final_results, f)
print("combined: ", final_results)

for k in final_results["results"]:
    data = final_results["results"][k]
    mean, std = np.mean(data), np.std(data)
    final_results["results"][k] = (mean, std)
with open(os.path.join(path, "errorbar.json"), "w") as f:
    json.dump(final_results, f)
print("error bar:", final_results)

df = pd.DataFrame(final_results["results"])
final_results_df = pd.DataFrame(df.iloc[0].apply(lambda x: f"{x:.3f}") + " $\pm$ " + df.iloc[1].apply(lambda x: f"{x:.3f}")).transpose()
final_results_df.to_csv(os.path.join(path, "final_results.csv"))