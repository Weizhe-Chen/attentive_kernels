from pathlib import Path
from urllib import request

from matplotlib import pyplot as plt
import numpy as np
import pypolo
from tqdm import tqdm

from parse_arguments import parse_arguments

plt.style.use(["nature", "science"])


class GPR(pypolo.models.GPR):
    """Gaussian Process Regression."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        kernel: pypolo.kernels.IKernel,
        noise: float,
        lr_hyper: float = 0.01,
        lr_nn: float = 0.001,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(x_train, y_train, kernel, noise, lr_hyper, lr_nn,
                         jitter)

    def optimize_once(self):
        self.train()
        self.opt_hyper.zero_grad()
        if self.opt_nn is not None:
            self.opt_nn.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        self.opt_hyper.step()
        if self.opt_nn is not None:
            self.opt_nn.step()
        self.eval()
        return loss.item()


class TrainingEvaluator:

    def __init__(self):
        self.log2pi = np.log(2 * np.pi)
        self.nums = []
        self.smses = []
        self.mslls = []
        self.nlpds = []
        self.rmses = []
        self.maes = []

    def eval_prediction(self, model, train_inputs, train_outputs):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        _, y_train = model.get_data()
        mean, std = model(train_inputs)
        error = np.fabs(mean - train_outputs)
        self.nums.append(model.num_train)
        self.smses.append(self.calc_smse(error))
        self.mslls.append(self.calc_msll(error, std, y_train))
        self.nlpds.append(self.calc_nlpd(error, std))
        self.rmses.append(self.calc_rmse(error))
        self.maes.append(self.calc_mae(error))
        return mean, std, error

    def calc_log_loss(self, error: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate negative log predictive density.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.
        std: np.ndarray, shape=(num_x * num_y, 1)
            Predictive standard deviation.

        Returns
        -------
        log_loss: np.ndarray, shape=(num_x * num_y, 1)
            negative log predictive density.

        """
        log_loss = 0.5 * self.log2pi + np.log(std) + 0.5 * np.square(
            error / std)
        return log_loss

    def calc_nlpd(self, error: np.ndarray, std: np.ndarray) -> np.float64:
        """Calculate mean negative log predictive density.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.
        std: np.ndarray, shape=(num_x * num_y, 1)
            Predictive standard deviation.

        Returns
        -------
        nlpd: np.float64
            Mean negative log predictive density.

        """
        nlpd = np.mean(self.calc_log_loss(error, std))
        return nlpd

    def calc_rmse(self, error: np.ndarray) -> np.float64:
        """Calculate root mean squared error.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.

        Returns
        -------
        rmse: np.float64
            Root mean squared error.

        """
        rmse = np.sqrt(np.mean(np.square(error)))
        return rmse

    def calc_mae(self, error: np.ndarray) -> np.float64:
        """Calculate mean absolute error.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.

        Returns
        -------
        mae: np.float64
            Mean absolute error.

        """
        mae = np.mean(error)
        return mae

    def calc_smse(self, error: np.ndarray) -> np.float64:
        """Calculate standardized mean squared error.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.

        Returns
        -------
        rmse: np.float64
            Standardized mean squared error.

        """
        mse = np.mean(np.square(error))
        smse = mse / self.train_outputs.var()
        return smse

    def calc_msll(
        self,
        error: np.ndarray,
        std: np.ndarray,
        y_train: np.ndarray,
    ) -> np.float64:
        """Calculate mean standardized log loss.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.
        std: np.ndarray, shape=(num_x * num_y, 1)
            Predictive standard deviation.
        y_train: np.ndarray, shap=(num_train, 1)
            Training targets.

        Returns
        -------
        msll: np.float64
            Mean standardized log predictive density.

        """
        log_loss = self.calc_log_loss(error, std)
        baseline = self.calc_log_loss(self.train_outputs - y_train.mean(),
                                      y_train.std())
        msll = np.mean(log_loss - baseline)
        return msll

    def save(self, save_dir: str) -> None:
        """Save all the metrics to the given output directory.

        Parameters
        ----------
        save_dir: str
            Directory to save file.

        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        table = np.column_stack((
            self.nums,
            self.smses,
            self.mslls,
            self.nlpds,
            self.rmses,
            self.maes,
        ))
        np.savetxt(
            f"{save_dir}/metrics.csv",
            table,
            fmt="%.8f",
            delimiter=',',
            header="samples,smse,msll,nlpd,rmse,mae",
        )
        print(f"Saved metrics.csv to {save_dir}")


def download_helens(data_path):
    path = Path(f"{data_path}/raw/mtsthelens_after.zip")
    if not path.is_file():
        print(f"Downloading to {path}...this step might take some time.")
        request.urlretrieve(
            url=
            "https://github.com/osrf/gazebo_tutorials/raw/master/dem/files/mtsthelens_after.zip",
            filename=path,
        )
        print("Done")
    import zipfile
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(f"{data_path}/raw/")


def get_environment(args, data_path):
    path = Path(f"{data_path}/preprocessed/{args.env_name}.npy")
    data_loader = pypolo.experiments.environments.DataLoader(str(path))
    return data_loader.get_data()


def get_sensor(args, env):
    sensor = pypolo.sensors.Sonar(
        rate=args.sensing_rate,
        env=env,
        env_extent=args.env_extent,
        noise_scale=args.noise_scale,
    )
    return sensor


def get_data(args, rng, sensor):
    """Returns the training data."""
    x1 = rng.uniform(args.task_extent[0],
                     args.task_extent[1],
                     size=args.num_train)
    x2 = rng.uniform(args.task_extent[2],
                     args.task_extent[3],
                     size=args.num_train)
    x_train = np.column_stack((x1, x2))
    y_train = sensor.sense(x_train)
    return x_train, y_train.reshape(-1, 1)


def get_model(args, x_init, y_init):
    """Returns the chosen model."""
    kernel = pypolo.experiments.utilities.get_kernel(args)
    model = GPR(
        x_train=x_init,
        y_train=y_init,
        kernel=kernel,
        noise=args.init_noise,
        lr_hyper=args.lr_hyper,
        lr_nn=args.lr_nn,
        jitter=args.jitter,
    )
    return model


def get_evaluator(args, sensor):
    evaluator = pypolo.experiments.Evaluator(
        sensor=sensor,
        task_extent=args.task_extent,
        eval_grid=args.eval_grid,
    )
    return evaluator


def main():
    args = parse_arguments(verbose=True)
    rng = pypolo.experiments.utilities.seed_everything(args.seed)
    if args.env_name == "helens":
        data_path = "../data/volcano/"
    else:
        data_path = "../data/srtm/"
    env = get_environment(args, data_path)
    sensor = get_sensor(args, env)
    x_train, y_train = get_data(args, rng, sensor)
    model = get_model(args, x_train, y_train)
    test_evaluator = get_evaluator(args, sensor)
    train_evaluator = TrainingEvaluator()
    pbar = tqdm(range(args.num_iter))
    losses = []
    for i in pbar:
        loss = model.optimize_once()
        losses.append(loss)
        pbar.set_description(f"Iter: {i:02d} loss: {loss: .2f}")
        test_evaluator.eval_prediction(model)
        train_evaluator.eval_prediction(model, *model.get_data())

    # Save outputs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test_evaluator.save(
        f"{args.output_dir}/{args.env_name}/{args.num_train}/{args.kernel}/test/"
    )
    train_evaluator.save(
        f"{args.output_dir}/{args.env_name}/{args.num_train}/{args.kernel}/train/"
    )

    # Save figures
    Path(f"{args.figure_dir}/{args.env_name}/{args.num_train}/").mkdir(
        parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=pypolo.experiments.visualizer.set_size(
        fraction=0.5))
    ax2 = ax1.twinx()
    color_loss, color_train, color_test = "tab:blue", "tab:green", "tab:red"
    ax1.plot(losses)
    ax1.set_ylabel("Negative Log Marginal Likelihood", color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax2.plot(train_evaluator.mslls, color=color_train, label="Training MSLL")
    ax2.plot(test_evaluator.mslls, color=color_test, label="Test MSLL")
    ax2.set_ylabel("Mean Standardized Log Loss")
    ax2.legend(loc="upper right")
    fig.savefig(
        f"{args.figure_dir}/{args.env_name}/{args.num_train}/{args.kernel}.pdf"
    )


if __name__ == "__main__":
    main()
