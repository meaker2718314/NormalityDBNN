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
df = []
kurtosis = []
N = 5000

for k in range(1, 21):
    print(f"Computing powers at a=0.05 for {k} degree(s) of freedom...")

    shap_wilk_FNR = 0
    average_kurtosis = 0

    for j in range(N):
        sample = np.random.standard_t(df=k, size=200)

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

    average_kurtosis /= N
    with torch.no_grad():
        x = torch.tensor(input_vectors, dtype=torch.float32)
        logit = model(x)
        p = torch.sigmoid(logit)

        p = torch.flatten(p).numpy()
        DBNN_FNR = np.sum(p > critical_value) / p.size
        powers_DBNN.append(np.round(1 - DBNN_FNR, 4))
        powers_SW.append(round(1 - shap_wilk_FNR / N, 4))
        df.append(k)
        kurtosis.append(average_kurtosis)


df = pd.DataFrame({
    "DF": df,
    "KURTOSIS": kurtosis,
    "POWER_DBNN": powers_DBNN,
    "POWER_SW": powers_SW
})

df.to_csv("DBNN_vs_SW_t_varied.csv", index=False)
