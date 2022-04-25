import torch
from pathlib import Path

import numpy as np
import pypolo
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from parse_arguments import parse_arguments


def rbf(dist, lengthscale):
    """Radial Basis Function."""
    return torch.exp(-0.5 * torch.square(dist / lengthscale))


class LengthscaleSelectionKernel(pypolo.kernels.AK):

    def __init__(self, amplitude, lengthscales, dim_input, dim_hidden,
                 dim_output):
        super().__init__(amplitude, lengthscales, dim_input, dim_hidden,
                         dim_output)

    def forward(self, x_1, x_2):
        dist = torch.cdist(x_1, x_2, p=2)
        repre1 = self.get_representations(x_1)
        repre2 = self.get_representations(x_2)
        cov_mat = 0.0
        for i in range(self.num_lengthscales):
            attention_lengthscales = torch.outer(repre1[:, i], repre2[:, i])
            cov_mat += rbf(dist, self.lengthscales[i]) * attention_lengthscales
        return self.amplitude * cov_mat


def get_data(data_path):
    x_train = np.load(f"{data_path}/x_train.npy")
    y_train = np.load(f"{data_path}/y_train.npy")
    x_test = np.load(f"{data_path}/x_test.npy")
    y_test = np.load(f"{data_path}/y_test.npy")
    return x_train, y_train, x_test, y_test


def get_kernel(args):
    if args.kernel == "rbf":
        kernel = pypolo.kernels.RBF(
            amplitude=args.init_amplitude,
            lengthscale=args.init_lengthscale,
        )
    elif args.kernel == "lengthscale_selection":
        kernel = LengthscaleSelectionKernel(
            amplitude=args.init_amplitude,
            lengthscales=np.linspace(
                args.min_lengthscale,
                args.max_lengthscale,
                args.dim_output,
            ),
            dim_input=args.dim_input,
            dim_hidden=args.dim_hidden,
            dim_output=args.dim_output,
        )
    else:
        raise ValueError(f"Kernel {args.kernel} is not supported.")
    return kernel


def get_model(args, x_train, y_train):
    kernel = get_kernel(args)
    model = pypolo.models.GPR(x_train=x_train,
                              y_train=y_train,
                              kernel=kernel,
                              noise=args.init_noise,
                              lr_hyper=args.lr_hyper,
                              lr_nn=args.lr_nn,
                              jitter=args.jitter,
                              is_normalized=False)
    return model


def optimize_model(args, model, verbose=True):
    model.optimize(num_iter=args.num_train_iter, verbose=verbose)


def plot_results(args, data, model, width=3.5 * 0.8):
    height = 0.4 * 0.618 * width
    fig, ax = plt.subplots(figsize=(width, height))
    plot_data(ax, data)
    plot_prediction(ax, data, model)
    plot_vlines(ax)
    save_plot(args, fig)


def plot_data(ax, data):
    x_train, y_train, x_test, y_test = data
    ax.plot(x_test, y_test, "r-", alpha=0.6, label="Target")
    ax.scatter(x_train,
               y_train,
               s=3,
               marker=".",
               color="k",
               alpha=0.8,
               label="Samples")
    ax.xaxis.set_ticklabels([])
    ax.set_xlim([x_test.min(), x_test.max()])
    ax.set_ylim([-2.5, 2.5])


def plot_prediction(ax, data, model):
    x_test = data[2]
    mean, std = model(x_test)
    ax.plot(x_test, mean, "b-", label="Prediction", alpha=0.6)
    ax.fill_between(x_test.ravel(),
                    mean.ravel() - 2.0 * std.ravel(),
                    mean.ravel() + 2.0 * std.ravel(),
                    color="b",
                    alpha=0.2,
                    label="Uncertainty")
    ax.set_ylabel(r"$y$")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%-.0f"))


def plot_vlines(ax):
    ax.vlines(np.arange(-0.6, 1.0, 0.4),
              -2.5,
              2.5,
              colors='k',
              linestyles='dashed',
              alpha=0.6)


def save_plot(args, fig, name=""):
    Path("./figures/").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.figure_dir}{args.kernel}{args.postfix}{name}.pdf",
                bbox_inches="tight")
    plt.close(fig)


def plot_vectors(ax, vectors, x_test):
    width = 2.1 / (len(x_test) - 1)
    bars = []
    bars.append(ax.bar(x_test, vectors[:, 0], width=width))
    bottom = vectors[:, 0].copy()
    for i in range(1, vectors.shape[1]):
        bars.append(ax.bar(
            x_test,
            vectors[:, i],
            width=width,
            bottom=bottom,
        ))
        bottom += vectors[:, i]
    ax.set_xlim([x_test.min(), x_test.max()])
    ax.set_ylim([0, 1])


def plot_weights(args, model, data, width=3.5 * 0.8):
    x_test = data[2]
    with torch.no_grad():
        w = model.kernel.get_representations(torch.tensor(x_test)).numpy()
        w /= w.sum(axis=1, keepdims=True)
    height = 0.4 * 0.618 * width
    fig, ax = plt.subplots(figsize=(width, height))
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel(r"$\mathbf{w}$")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%-.0f"))
    plot_vectors(ax, w, x_test.ravel())
    plot_vlines(ax)
    save_plot(args, fig, name="_weights")


def main():
    args = parse_arguments()
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)
    pypolo.experiments.utilities.seed_everything(args.seed)
    data = get_data("../data/sin/")
    model = get_model(args, *data[:2])
    optimize_model(args, model, verbose=True)
    plot_results(args, data, model)
    if args.kernel == "lengthscale_selection":
        plot_weights(args, model, data)
    print(f"Saved to figures/")


if __name__ == "__main__":
    main()
