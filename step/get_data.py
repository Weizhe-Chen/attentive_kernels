from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from pypolo.utilities.scalers import MinMaxScaler, StandardScaler

# Demo configurations
num_trains = [30, 30, 50, 30, 30]
eps = 0.0
num_test = 300
offset = 4
xmax = 5
noise_scale = 0.1


# Ground-truth function
def fn(x):
    y = []
    for each in x:
        if 0.0 <= each < 1.0:
            y.append(np.sin(10 * each) - offset)
        elif 1.0 <= each < 2.0:
            y.append(np.sin(10 * each) + offset)
        elif 2.0 <= each < 3.0:
            y.append(3 * np.sin(40 * each) - offset)
        elif 3.0 <= each < 4.0:
            y.append(np.sin(10 * each) + offset)
        elif 4.0 <= each <= 5.0:
            y.append(np.sin(10 * each) - offset)
        else:
            raise ValueError("Input is out of range [0, 5]")
    return np.vstack(y)


# Training set
rng = np.random.RandomState(0)
x_train = np.hstack([
    rng.uniform(low=low + eps, high=low + 1 - eps, size=num_trains[i])
    for low, i in enumerate(range(5))
] + [x - 1e-2 for x in range(1, 5)] + [x + 1e-2 for x in range(1, 5)])
x_train = x_train.reshape(-1, 1)
y_train = fn(x_train)

# Test set
x_test = np.linspace(0, xmax, num_test).reshape(-1, 1)
y_test = fn(x_test)

# Standardize datasets
minmaxer = MinMaxScaler(x_train, (-1, 1))
standardizer = StandardScaler(y_train)
x_train = minmaxer.preprocess(x_train)
x_test = minmaxer.preprocess(x_test)
y_train = standardizer.preprocess(y_train)
y_test = standardizer.preprocess(y_test)

# Add some observational noise to the training outputs
y_train += noise_scale * rng.randn(len(x_train), 1)

# Visualization
width = 3.5
height = 0.4 * 0.618 * width
fig, ax = plt.subplots(figsize=(width, height))
ax.set_xlim([x_test.min(), x_test.max()])
ax.set_ylim([-2.5, 2.5])
ax.xaxis.set_ticklabels([])
ax.set_ylabel(f"$y$")
ax.plot(x_test, y_test, "r-", alpha=0.6, label="Target")
ax.scatter(x_train,
           y_train,
           s=3,
           marker=".",
           color="k",
           alpha=0.8,
           label="Samples")
ax.vlines(np.arange(-0.6, 1.0, 0.4),
          -2.5,
          2.5,
          colors='k',
          linestyles='dashed',
          alpha=0.6)
Path("./figures/").mkdir(parents=True, exist_ok=True)
fig.savefig(f"./figures/data.pdf", bbox_inches="tight")
plt.close(fig)

# Saving
data_path = "../data/step/"
Path(data_path).mkdir(parents=True, exist_ok=True)
np.save(f"{data_path}/x_train.npy", x_train)
np.save(f"{data_path}/y_train.npy", y_train)
np.save(f"{data_path}/x_test.npy", x_test)
np.save(f"{data_path}/y_test.npy", y_test)
print("Data saved to ./data/")
