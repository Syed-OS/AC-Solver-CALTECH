import pandas as pd
from ac_solver.search.greedy import greedy_search
from ac_solver.data.load_instances import load_subset_instances  
import time

MAX_NODES = 250_000
ALPHA = 1.0
BETA = 0.0  
instances = load_subset_instances()  

results = []
total_start = time.time()

for instance_id, presentation in instances:
    start = time.time()

    solved, path = greedy_search(
        presentation=presentation,
        max_nodes_to_explore=MAX_NODES,
        alpha=ALPHA,
        beta=BETA
    )

    runtime = time.time() - start

    results.append({
        "instance_id": instance_id,
        "solved": int(solved),
        "runtime_sec": runtime
    })

total_runtime = time.time() - total_start

df = pd.DataFrame(results)
df.to_csv("results/greedy_250k_baseline.csv", index=False)

print("Total solves:", df["solved"].sum())
print("Total runtime:", total_runtime)