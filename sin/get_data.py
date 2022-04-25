from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from pypolo.utilities.scalers import MinMaxScaler, StandardScaler

# Demo configurations
num_train = 40
num_test = 300
noise_scale = 0.1
xmin = 0
xmax = 1


# Ground-truth function
def fn(x):
    return x * np.sin(40 * x**4)


# Training set
rng = np.random.RandomState(0)
x_train = np.hstack([
    rng.uniform(low=xmin, high=2 * xmax / 3, size=num_train),
    rng.uniform(low=2 * xmax / 3, high=xmax, size=2 * num_train)
])
x_train = x_train.reshape(-1, 1)
y_train = fn(x_train)

# Test set
x_test = np.linspace(xmin, xmax, num_test).reshape(-1, 1)
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
Path("./figures/").mkdir(parents=True, exist_ok=True)
fig.savefig(f"./figures/data.pdf", bbox_inches="tight")
plt.close(fig)

# Saving
data_path = "../data/sin/"
Path(data_path).mkdir(parents=True, exist_ok=True)
np.save(f"{data_path}/x_train.npy", x_train)
np.save(f"{data_path}/y_train.npy", y_train)
np.save(f"{data_path}/x_test.npy", x_test)
np.save(f"{data_path}/y_test.npy", y_test)
print("Data saved to ./data/")
