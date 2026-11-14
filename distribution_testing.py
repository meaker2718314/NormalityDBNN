import torch
import numpy as np
import pandas as pd
from scipy import stats
from models import DBNN


model = DBNN(22, [100, 20, 10])
model.load_state_dict(torch.load("DBNN2trained.pth", weights_only=True))
model.eval()

np.random.seed(123456)

input_vectors = []

null_quantile = pd.read_csv("testing_data/nullDistributionQuantiles.csv", index_col='SAMPLE_SIZE')

# Convert to dictionary where keys are the IDs
null_quantile = null_quantile.squeeze().to_dict(orient='index')
sample_sizes = sorted(null_quantile.keys())

for key in sample_sizes:  # Make a simple dictionary instead of a nested dictionary, extract the relevant quantiles
    null_quantile[key] = null_quantile[key]['CRITICAL_ALPHA_05']

powers_SW = []
powers_DBNN = []
N = 5000

for size in sample_sizes:
    print(f"Computing powers at a=0.05 for sample size {size}...")
    shap_wilk_FNR = 0
    for j in range(N):

        # The distribution below can be changed as desired
        a = np.random.choice([0.5, 1.0, 1.5])
        b = np.random.choice([0.5, 1.0, 1.5])
        sample = np.random.beta(a=a, b=b, size=size)

        standardized_sample = (sample - np.mean(sample)) / np.std(sample)

        # Build the input vector
        quantiles = [np.quantile(standardized_sample, q=k/20) for k in range(1, 20)]
        minimum = np.min(standardized_sample)
        maximum = np.max(standardized_sample)

        input_vector = quantiles + [minimum] + [maximum] + [size]
        input_vectors.append(input_vector)
        stat, p_val = stats.shapiro(standardized_sample)
        shap_wilk_FNR += int(p_val > 0.05)

    with torch.no_grad():
        x = torch.tensor(input_vectors, dtype=torch.float32)
        logit = model(x)
        p = torch.sigmoid(logit)

    p = torch.flatten(p).numpy()
    DBNN_FNR = np.sum(p > null_quantile[size]) / p.size
    powers_DBNN.append(np.round(1 - DBNN_FNR, 4))
    powers_SW.append(round(1 - shap_wilk_FNR / N, 4))

df = pd.DataFrame({
    "SAMPLE_SIZE": sample_sizes,
    "POWER_DBNN": powers_DBNN,
    "POWER_SW": powers_SW
})

df.to_csv("DBNN_vs_SW_beta.csv", index=False)
