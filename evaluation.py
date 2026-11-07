import torch
import pandas as pd
from models import DBNN
import numpy as np

model = DBNN(22, [100, 20, 10])
model.load_state_dict(torch.load("DBNN2trained.pth", weights_only=True))
model.eval()

null_quantile = pd.read_csv("testing_data/nullDistributionQuantiles.csv", index_col='SAMPLE_SIZE')
# Convert to dictionary where keys are the IDs
null_quantile = null_quantile.squeeze().to_dict(orient='index')
sample_sizes = sorted(null_quantile.keys())

# Replace with desired sample of size 10 to 500
sample = np.array([0, 1, 1, 1, 1, 2, 2, 2, 3, 5, 8, 8, 11, 14, 15, 20, 22])


input_size = int(sample.size)
if input_size < 10 or input_size > 500:
    print("Sample size is not contained in [10, 500], test may be misleading...")

standardized_sample = (sample - np.mean(sample)) / np.std(sample)

# Build the input vector
quantiles = [np.quantile(standardized_sample, q=k / 20) for k in range(1, 20)]
minimum = np.min(standardized_sample)
maximum = np.max(standardized_sample)

input_vector = quantiles + [minimum] + [maximum] + [input_size]

while input_size not in sample_sizes:
    input_size -= 1  # We find the closest smaller sample size which has known cutoffs,
    # This is the more conservative choice, as opposed to rounding up.

quantile_05 = null_quantile[input_size]['CRITICAL_ALPHA_05']
quantile_10 = null_quantile[input_size]['CRITICAL_ALPHA_10']
with torch.no_grad():
    x = torch.tensor(input_vector, dtype=torch.float32)
    logit = model(x)
    p = torch.sigmoid(logit)

print(f"Model confidence in H0: {p.numpy()[0]:.5f}")

if p > quantile_10:
    print("Fail to reject normality at 0.05 and 0.10 significance level...")
elif quantile_05 < p <= quantile_10:
    print("Fail to reject normality at 0.05 significance level, reject normality at 0.10 significance level...")
else:
    print("Reject normality at both 0.05 and 0.10 significance level...")
