import os
import json
import numpy as np
import scipy.stats as st
import pandas as pd
import sys

result_dir = "./results"
path = sys.argv[1]
filename = sys.argv[2]
id = f"{path}_{filename}"

conf = 0.95
exps = [d for d in os.listdir(os.path.join(result_dir, path)) if "SEED" in d]
print(f"exps: {exps}")

final_results = {
    "seeds": exps,
    "results": {},
}

for d in exps:
    with open(os.path.join(result_dir, path, d, f"{id}.json"), 'r') as f:
        result = json.load(f)

    for k in result:
        if k in final_results["results"]:
            final_results["results"][k].append(result[k])
        else:
            final_results["results"][k] = [result[k]]

# combined results
with open(os.path.join(result_dir, path, f"{id}_combined_results.json"), "w") as f:
    json.dump(final_results, f)

print("=== Individual Results ===")
print("F1", final_results["results"]["F1"])
print("F1_PA", final_results["results"]["F1_PA"])
print("AUROC", final_results["results"]["ROC_AUC"])
print("AUPRC", final_results["results"]["PR_AUC"])


for k in final_results["results"]:
    data = final_results["results"][k]
    mean, std = np.mean(data), np.std(data)
    final_results["results"][k] = (mean, std)

# error bar
with open(os.path.join(result_dir, path, f"{id}_errorbar.json"), "w") as f:
    json.dump(final_results, f)

df = pd.DataFrame(final_results["results"])
final_results_df = pd.DataFrame(df.iloc[0].apply(lambda x: f"{x:.3f}") + " $\pm$ " + df.iloc[1].apply(lambda x: f"{x:.3f}")).transpose()
final_results_df.to_csv(os.path.join(result_dir, path, f"{id}_final_results.csv"))


print("=== Combined Results ===")
print("F1", final_results["results"]["F1"])
print("F1_PA", final_results["results"]["F1_PA"])
print("AUROC", final_results["results"]["ROC_AUC"])
print("AUPRC", final_results["results"]["PR_AUC"])
