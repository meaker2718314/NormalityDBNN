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

null_quantile = null_quantile.squeeze().to_dict(orient='index')
critical_value = null_quantile[200]['CRITICAL_ALPHA_05']

powers_SW = []
powers_DBNN = []
kurtosis = []
N = 1000

for k in range(1, 160):
    if k % 5 == 0:
        print(f"Computing powers at a=0.05 for distribution #{k} ...")
    shap_wilk_FNR = 0
    average_kurtosis = 0

    scale2 = np.random.uniform(1.15, 4)
    p = np.random.uniform(0.05, 0.2)

    for j in range(N):
        sample_1 = np.random.normal(0, 1, size=200)
        sample_2 = np.random.normal(0, scale2, size=200)
        inclusion = np.random.binomial(p=p, n=1, size=200)
        sample = np.where(inclusion, sample_2, sample_1)
        standardized_sample = (sample - np.mean(sample)) / np.std(sample)
        average_kurtosis += stats.kurtosis(standardized_sample)
        # Build the input vector
        quantiles = [np.quantile(standardized_sample, q=k/20) for k in range(1, 20)]
        minimum = np.min(standardized_sample)
        maximum = np.max(standardized_sample)

        input_vector = quantiles + [minimum] + [maximum] + [200]
        input_vectors.append(input_vector)
        stat, p_val = stats.shapiro(standardized_sample)
        shap_wilk_FNR += int(p_val > 0.05)

    average_kurtosis = round(average_kurtosis / N, 4)
    with torch.no_grad():
        x = torch.tensor(input_vectors, dtype=torch.float32)
        logit = model(x)
        p = torch.sigmoid(logit)

        p = torch.flatten(p).numpy()
        DBNN_FNR = np.sum(p > critical_value) / p.size
        powers_DBNN.append(np.round(1 - DBNN_FNR, 3))
        powers_SW.append(round(1 - shap_wilk_FNR / N, 3))
        kurtosis.append(average_kurtosis)


df = pd.DataFrame({
    "KURTOSIS": kurtosis,
    "POWER_DBNN": powers_DBNN,
    "POWER_SW": powers_SW
})

df.to_csv("DBNN_vs_SW_mixed_kurtosis.csv", index=False)
