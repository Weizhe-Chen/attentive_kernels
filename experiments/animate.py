import pypolo
from pypolo.experiments import visualizer
import numpy as np


def main():
    args = pypolo.experiments.argparser.parse_arguments()
    experiment_id = "/".join([
        str(args.seed),
        args.env_name,
        args.strategy,
        args.kernel,
    ])
    save_dir = args.output_dir + experiment_id
    log = np.load(f"{save_dir}/log.npz")
    axes, caxes = visualizer.get_matplotlib_axes()
    visualizer.plot_image(
        args,
        axes[0, 0],
        caxes[0, 0],
        log["eval_outputs"],
        "Ground Truth",
    )
    snapshot_interval = 1 if args.strategy == "myopic" else 5
    for index, num in enumerate(log["nums"]):
        if index % snapshot_interval != 0:
            continue
        visualizer.clear_axes(axes, caxes)
        visualizer.set_limits(axes, args)
        visualizer.plot_image(args, axes[0, 1], caxes[0, 1],
                              log["means"][:, index], "Prediction")
        visualizer.plot_image(args, axes[1, 0], caxes[1, 0],
                              log["stds"][:, index], "Uncertainty")
        visualizer.plot_image(args, axes[1, 1], caxes[1, 1],
                              log["errors"][:, index], "Error")
        axes[1, 0].scatter(log["xs"][:num, 0],
                           log["xs"][:num, 1],
                           marker='.',
                           color='k',
                           alpha=0.6)
        axes[1, 0].scatter(log["xs"][num - 1, 0],
                           log["xs"][num - 1, 1],
                           marker='*',
                           color='w',
                           alpha=0.9,
                           s=200)
        visualizer.pause()


if __name__ == "__main__":
    main()
