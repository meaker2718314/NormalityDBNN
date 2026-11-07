import torch
import numpy as np
import pandas as pd
from models import DBNN

model = DBNN(22, [100, 20, 10])
model.load_state_dict(torch.load("DBNN2trained.pth", weights_only=True))
model.eval()

sample_sizes = [5*i for i in range(2, 11)]
sample_sizes.extend(10*i for i in range(6, 51))

ss_ref = []
critical_vals_05 = []
critical_vals_10 = []

for size in sample_sizes:
    print(f"Computing quantiles for sample size {size}")
    input_vectors = []
    for j in range(10000):
        sample = np.random.normal(size=size)
        standardized_sample = (sample - np.mean(sample)) / np.std(sample)

        # Build the input vector
        quantiles = [np.quantile(standardized_sample, q=k/20) for k in range(1, 20)]
        minimum = np.min(standardized_sample)
        maximum = np.max(standardized_sample)

        input_vector = quantiles + [minimum] + [maximum] + [size]
        input_vectors.append(input_vector)

    with torch.no_grad():
        x = torch.tensor(input_vectors, dtype=torch.float32)
        logit = model(x)
        p_normal = torch.sigmoid(logit)

    p_normal = torch.flatten(p_normal).numpy()
    crit_05 = np.round(np.quantile(p_normal, 0.05), 4)
    crit_10 = np.round(np.quantile(p_normal, 0.10), 4)
    ss_ref.append(size)
    critical_vals_05.append(crit_05)
    critical_vals_10.append(crit_10)

df = pd.DataFrame({
    "SAMPLE_SIZE": ss_ref,
    "CRITICAL_ALPHA_05": critical_vals_05,
    "CRITICAL_ALPHA_10": critical_vals_10,
})

df.to_csv("nullDistributionQuantiles.csv", index=False)

