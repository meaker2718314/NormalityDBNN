import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import DBNN
from distributions import training_sample_distribution, validation_sample_distribution


model1 = DBNN(22, [100, 20, 10])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)

np.random.seed(123456)  # Set random seed for reproducibility

sample_sizes = []

for s in range(9):
    sample_sizes.extend(4000 * [5 * s + 10])

size = 55
while size <= 500:
    sample_sizes.extend(1000 * [size])
    size += 5

sample_sizes = np.array(sample_sizes)
np.random.shuffle(sample_sizes)

sample_sizes_val = []
for s in range(9):
    sample_sizes_val.extend(800 * [5 * s + 10])

size = 55
while size <= 500:
    sample_sizes_val.extend(200 * [size])
    size += 5

sample_sizes_val = np.array(sample_sizes_val)
np.random.shuffle(sample_sizes_val)

validation_response = []
validation_input = []
for size in sample_sizes_val:
    sample, isNormal = validation_sample_distribution(size)  # (np.array, bool)

    validation_response.append(int(isNormal))

    standardized_sample = (sample - np.mean(sample)) / np.std(sample)

    # Build the input vector
    quantiles = [np.quantile(standardized_sample, q=k / 20) for k in range(1, 20)]
    minimum = np.min(standardized_sample)
    maximum = np.max(standardized_sample)

    input_vector = quantiles + [minimum] + [maximum] + [size]

    validation_input.append(input_vector)

X_val = np.array(validation_input)
Y_val = np.array(validation_response)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

dataset_val = TensorDataset(X_val, Y_val)
validation_loader = DataLoader(dataset_val, batch_size=128, shuffle=False)  # Order of evaluation doesn't matter...

epoch = 1  # Each epoch consists of entirely novel training data, i.e. "online learning"
# This will make explicit overfitting a near impossibility, since a particular piece of data is only even seen once...

while True:  # Train indefinitely until user input specifies otherwise
    for i in range(12):  # Partition each epoch to limit memory usage when computing and storing sample vectors...
        training_input = []
        training_response = []
        ss = sample_sizes[len(sample_sizes) // 12 * i: len(sample_sizes) // 12 * (i + 1)].tolist()
        for size in ss:
            sample, isNormal = training_sample_distribution(size)  # (np.array, bool)

            training_response.append(int(isNormal))

            standardized_sample = (sample - np.mean(sample)) / np.std(sample)

            # Build the input vector
            quantiles = [np.quantile(standardized_sample, q=k / 20) for k in range(1, 20)]
            minimum = np.min(standardized_sample)
            maximum = np.max(standardized_sample)

            input_vector = quantiles + [minimum] + [maximum] + [size]

            training_input.append(input_vector)

        X = np.array(training_input)
        Y = np.array(training_response)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X, Y)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        model1.train()
        average_loss = 0

        print(f"Starting training for Epoch {epoch}, Partition {i}...")

        for xb, yb in train_loader:
            prediction = model1(xb)
            loss = criterion(prediction, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss += loss.item()

        print(f"Results for Epoch {epoch}, Partition {i}: Training BCE Loss ="
              f" {average_loss / len(train_loader):.6f}")

        val_loss = 0
        model1.eval()
        with torch.no_grad():
            for xb, yb in validation_loader:
                loss = criterion(model1(xb), yb)
                val_loss += loss.item()

        print(f"Results for Epoch {epoch}, Partition {i}: Validation BCE loss = "
              f"{val_loss / len(validation_loader):.6f}")

    epoch += 1

    if input("Continue training (Y/N)? ") == "N":
        break

torch.save(model1.state_dict(), 'DBNN2trained.pth')
