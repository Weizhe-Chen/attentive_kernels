import torch
from torch.nn.parameter import Parameter
from pathlib import Path
from time import time
import pypolo
import numpy as np


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


class InstanceSelectionKernel(pypolo.kernels.IKernel):

    def __init__(self, amplitude, lengthscale, dim_input, dim_hidden,
                 dim_output):
        super().__init__(amplitude)
        self.__free_lengthscale = Parameter(
            pypolo.utilities.linalg.unconstraint(lengthscale))
        self.nn = pypolo.kernels.TwoHiddenLayerTanhNN(
            dim_input,
            dim_hidden,
            dim_output,
        ).double()

    @property
    def lengthscale(self):
        return pypolo.utilities.linalg.constraint(self.__free_lengthscale)

    @lengthscale.setter
    def lengthscale(self, lengthscale):
        with torch.no_grad():
            self.__free_lengthscale.copy_(
                pypolo.utilities.linalg.unconstraint(lengthscale))

    def get_representations(self, x):
        z = self.nn(x)
        representations = z / z.norm(dim=1, keepdim=True)
        return representations

    def forward(self, x_1, x_2):
        dist = torch.cdist(x_1, x_2, p=2)
        repre1 = self.get_representations(x_1)
        repre2 = self.get_representations(x_2)
        attention_inputs = repre1 @ repre2.t()
        cov_mat = self.amplitude * attention_inputs * rbf(
            dist, self.lengthscale)
        return cov_mat


class NNx2(pypolo.kernels.IKernel):

    def __init__(self, amplitude, lengthscales, dim_input, dim_hidden,
                 dim_output):
        super().__init__(amplitude)
        self.lengthscales = lengthscales
        self.nn_mask = pypolo.kernels.TwoHiddenLayerTanhNN(
            dim_input,
            dim_hidden,
            dim_output,
        ).double()
        self.nn_weight = pypolo.kernels.TwoHiddenLayerTanhNN(
            dim_input,
            dim_hidden,
            dim_output,
        ).double()

    @property
    def num_lengthscales(self):
        return len(self.lengthscales)

    def get_representations_mask(self, x):
        z = self.nn_mask(x)
        representations = z / z.norm(dim=1, keepdim=True)
        return representations

    def get_representations_weight(self, x):
        z = self.nn_weight(x)
        representations = z / z.norm(dim=1, keepdim=True)
        return representations

    def forward(self, x_1, x_2):
        dist = torch.cdist(x_1, x_2, p=2)
        weight_1 = self.get_representations_weight(x_1)
        weight_2 = self.get_representations_weight(x_2)
        cov_mat = 0.0
        for i in range(self.num_lengthscales):
            attention_lengthscales = torch.outer(
                weight_1[:, i],
                weight_2[:, i],
            )
            cov_mat += rbf(dist, self.lengthscales[i]) * attention_lengthscales
        mask_1 = self.get_representations_mask(x_1)
        mask_2 = self.get_representations_mask(x_2)
        attention_inputs = mask_1 @ mask_2.t()
        cov_mat *= self.amplitude * attention_inputs
        return cov_mat


def get_sensor(args, env):
    sensor = pypolo.sensors.Sonar(
        rate=args.sensing_rate,
        env=env,
        env_extent=args.env_extent,
        noise_scale=args.noise_scale,
    )
    return sensor


def get_pilot_data(args, rng, sensor):
    bezier = pypolo.strategies.Bezier(task_extent=args.task_extent, rng=rng)
    x_init = bezier.get(num_states=args.num_init_samples)
    y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)
    return x_init, y_init


def get_robot(x_init, args):
    robot = pypolo.robots.USV(
        init_state=np.array([x_init[-1, 0], x_init[-1, 1], np.pi / 2]),
        control_rate=args.control_rate,
        max_lin_vel=args.max_lin_vel,
        tolerance=args.tolerance,
        sampling_rate=args.sensing_rate,
    )
    return robot


def get_kernel(args):
    if args.kernel == "ak":
        kernel = pypolo.kernels.AK(
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
    elif args.kernel == "nnx2":
        kernel = NNx2(
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
    elif args.kernel == "instance_selection":
        kernel = InstanceSelectionKernel(
            amplitude=args.init_amplitude,
            lengthscale=args.init_lengthscale,
            dim_input=args.dim_input,
            dim_hidden=args.dim_hidden,
            dim_output=args.dim_output,
        )
    else:
        raise ValueError(f"Kernel {args.kernel} is not supported.")
    return kernel


def get_model(args, x_init, y_init):
    kernel = get_kernel(args)
    model = pypolo.models.GPR(
        x_train=x_init,
        y_train=y_init,
        kernel=kernel,
        noise=args.init_noise,
        lr_hyper=args.lr_hyper,
        lr_nn=args.lr_nn,
        jitter=args.jitter,
    )
    model.optimize(num_iter=model.num_train, verbose=False)
    return model


def get_evaluator(args, sensor):
    evaluator = pypolo.experiments.Evaluator(
        sensor=sensor,
        task_extent=args.task_extent,
        eval_grid=args.eval_grid,
    )
    return evaluator


def get_strategy(args, rng, robot):
    """Get sampling strategy."""
    if args.strategy == "random":
        return pypolo.strategies.RandomSampling(
            task_extent=args.task_extent,
            rng=rng,
        )
    elif args.strategy == "active":
        return pypolo.strategies.ActiveSampling(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
        )
    elif args.strategy == "myopic":
        return pypolo.strategies.MyopicPlanning(
            task_extent=args.task_extent,
            rng=rng,
            num_candidates=args.num_candidates,
            robot=robot,
        )
    else:
        raise ValueError(f"Strategy {args.strategy} is not supported.")


def run(args, rng, model, strategy, sensor, evaluator):
    while model.num_train < args.max_num_samples:
        x_new = strategy.get(model=model)
        y_new = sensor.sense(x_new, rng).reshape(-1, 1)
        model.add_data(x_new, y_new)
        model.optimize(num_iter=len(y_new), verbose=False)
        evaluator.eval_prediction(model)
        pypolo.experiments.utilities.print_metrics(model, evaluator)


def save(args, evaluator):
    print("Saving metrics......")
    experiment_id = "/".join([
        str(args.seed),
        args.env_name,
        args.strategy,
        args.kernel + args.postfix,
    ])
    save_dir = args.output_dir + experiment_id
    evaluator.save(save_dir)


def main():
    args = pypolo.experiments.argparser.parse_arguments()
    rng = pypolo.experiments.utilities.seed_everything(args.seed)
    data_path = "../data/srtm"
    Path(data_path).mkdir(exist_ok=True, parents=True)
    env = pypolo.experiments.environments.get_environment(
        args.env_name, data_path)
    sensor = get_sensor(args, env)
    x_init, y_init = get_pilot_data(args, rng, sensor)
    robot = get_robot(x_init, args)
    model = get_model(args, x_init, y_init)
    evaluator = get_evaluator(args, sensor)
    evaluator.eval_prediction(model)
    strategy = get_strategy(args, rng, robot)
    start = time()
    run(args, rng, model, strategy, sensor, evaluator)
    end = time()
    save(args, evaluator)
    print(f"Time used: {end - start:.1f} seconds")


if __name__ == "__main__":
    main()
