import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from skimage import transform
from torch.nn.parameter import Parameter
from tqdm import tqdm

# Matplotlib settings
plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"

# Kernel settings
init_amplitude = 1.0
init_lengthscale = 0.5
init_noise = 1.0
lr_hyper = 0.01
lr_nn = 0.001
dim_input = 2
dim_hidden = 10
dim_output = 10
min_lengthscale = 0.01
max_lengthscale = 0.5
jitter = 1e-6

# Experiment settings
max_num_samples = 700
num_init_samples = 50
noise_scale = 1.0
tolerance = 0.1
max_lin_vel = 1.0
control_rate = 10.0
sensing_rate = 1.0
eval_grid = [50, 50]
env_extent = [-11.0, 11.0, -11.0, 11.0]
task_extent = [-10.0, 10.0, -10.0, 10.0]
num_candidates = 1000
seed = 0


def softplus(x):
    return F.softplus(x, 1.0, 20.0) + 1e-6


def inv_softplus(y):
    if torch.any(y <= 0.0):
        raise ValueError("Input to `inv_softplus` must be positive.")
    _y = y - 1e-6
    return _y + torch.log(-torch.expm1(-_y))


def constraint(free_parameter):
    return softplus(free_parameter)


def unconstraint(parameter):
    return inv_softplus(torch.tensor(
        parameter,
        dtype=torch.float64,
    ))


def robust_cholesky(cov_mat, jitter=1e-6, num_attempts=3):
    L, info = torch.linalg.cholesky_ex(cov_mat, out=None)
    if not torch.any(info):
        return L
    if torch.any(torch.isnan(cov_mat)):
        raise ValueError("Encountered NaN in cov_mat.")
    _cov_mat = cov_mat.clone()
    jitter_prev = 0.0
    jitter_new = jitter
    for i in range(num_attempts):
        is_positive_definite = info > 0
        jitter_new = jitter * (10**i)
        increment = is_positive_definite * (jitter_new - jitter_prev)
        _cov_mat.diagonal().add_(increment)
        jitter_prev = jitter_new
        print("Matrix is not positive definite! " +
              f"Added {jitter_new:.1e} to the diagonal.")
        L, info = torch.linalg.cholesky_ex(_cov_mat, out=None)
        if not torch.any(info):
            return L
    raise ValueError(
        "Covariance matrix is still not positive-definite " +
        f"after adding {jitter_new:.1e} to the diagonal elements.")


class GridMap:

    def __init__(self, matrix, extent):
        self.matrix = matrix
        self.extent = extent
        self.num_rows, self.num_cols = matrix.shape
        self.eps = 1e-4
        self.x_cell_size = (extent[1] - extent[0]) / self.num_cols + self.eps
        self.y_cell_size = (extent[3] - extent[2]) / self.num_rows + self.eps

    def xs_to_cols(self, xs):
        cols = ((xs - self.extent[0]) / self.x_cell_size).astype(int)
        return cols

    def ys_to_rows(self, ys):
        rows = ((ys - self.extent[2]) / self.y_cell_size).astype(int)
        return rows

    def get(self, xs, ys):
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        values = self.matrix[rows, cols]
        return values

    def set(self, xs, ys, values) -> None:
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        self.matrix[rows, cols] = values


class Sonar:

    def __init__(self, rate, env, env_extent, noise_scale) -> None:
        if rate <= 0.0:
            raise ValueError("rate must be positive.")
        self.dt = 1.0 / rate
        self.env = GridMap(env, env_extent)
        self.noise_scale = noise_scale

    def sense(self, states, rng=None):
        if states.ndim == 1:
            states = states.reshape(1, -1)
        observations = self.env.get(states[:, 0], states[:, 1])
        if rng is not None:
            observations = rng.normal(loc=observations, scale=self.noise_scale)
        return observations


class DubinsCar:

    def __init__(self, rate):
        self.rate = rate
        self.dt = 1.0 / rate

    @staticmethod
    def check_dtype(state, action):
        if state.dtype != np.float64:
            raise TypeError("state.dtype should be np.float64")
        if action.dtype != np.float64:
            raise TypeError("action.dtype should be np.float64")

    @staticmethod
    def normalize_angle(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def step(self, state, action):
        self.check_dtype(state, action)
        x, y, o = state
        v, w = action
        state[0] = x + v * np.cos(o) * self.dt
        state[1] = y + v * np.sin(o) * self.dt
        state[2] = o + w * self.dt
        state[2] = self.normalize_angle(state[2])
        return state


class USV:

    def __init__(self, init_state, control_rate, max_lin_vel, tolerance,
                 sampling_rate):
        dynamics = DubinsCar(control_rate)
        self.state = init_state
        self.tolerance = tolerance
        self.sampling_locations = []
        self.goal_states = []
        self.dynamics = dynamics
        self.__cumulative_time = 0.0
        if sampling_rate <= 0.0:
            raise ValueError("Sampling rate must be positive.")
        self.sampling_dt = 1.0 / sampling_rate
        self.max_lin_vel = max_lin_vel

    def control(self):
        x, y, o = self.state
        # Compute distance to the goal.
        goal_state = self.goal_states[0]
        goal_x, goal_y = goal_state[:2]
        x_diff = goal_x - x
        y_diff = goal_y - y
        dist = np.hypot(x_diff, y_diff)
        # Compute the goal position in the odometry frame.
        x_odom = np.cos(o) * x_diff + np.sin(o) * y_diff
        y_odom = -np.sin(o) * x_diff + np.cos(o) * y_diff
        linear_velocity = self.max_lin_vel * np.tanh(x_odom)
        # angular proportional parameter is set to 2.0
        angular_velocity = 2.0 * np.arctan2(y_odom, x_odom)
        action = np.array([linear_velocity, angular_velocity])
        return dist, action

    @property
    def has_goal(self):
        return len(self.goal_states) > 0

    def update(self, dist, action):
        # Update state
        self.state = self.dynamics.step(self.state, action)
        # Get sensing observation at a fixed rate.
        self.__cumulative_time += self.dynamics.dt
        if self.__cumulative_time > self.sampling_dt:
            self.sampling_locations.append([
                self.state[0],
                self.state[1],
            ])
            self.__cumulative_time = 0.0
        # Delete the first goal if we already achieved it.
        if self.has_goal and (dist < self.tolerance):
            self.goal_states = self.goal_states[1:]

    def commit_data(self):
        x_new = np.vstack(self.sampling_locations)
        self.sampling_locations = []
        return x_new


class StandardScaler:

    def __init__(self, values):
        if values.ndim != 2:
            raise ValueError("values.shape=(num_samples, num_dims)")
        self.scale = values.std(axis=0, keepdims=True)
        if np.any(self.scale <= 0.0):
            raise ValueError("scale must be positive")
        self.mean = values.mean(axis=0, keepdims=True)

    def preprocess(self, raw):
        transformed = (raw - self.mean) / self.scale
        return transformed

    def postprocess_mean(self, transformed):
        raw = transformed * self.scale + self.mean
        return raw

    def postprocess_std(self, transformed):
        raw = transformed * self.scale
        return raw


class MinMaxScaler:

    def __init__(self, values, expected_range=(-1.0, 1.0)):
        self.min = expected_range[0]
        self.max = expected_range[1]
        # `ptp` is the acronym for ‘peak to peak’.
        self.ptp = expected_range[1] - expected_range[0]
        if self.ptp <= 0.0:
            raise ValueError("Expected range must be positive.")
        self.data_min = values.min(axis=0, keepdims=True)
        self.data_max = values.max(axis=0, keepdims=True)
        self.data_ptp = self.data_max - self.data_min
        if np.any(self.data_ptp <= 0.0):
            raise ValueError("Data range must be positive.")

    def preprocess(self, raw):
        standardized = (raw - self.data_min) / self.data_ptp
        transformed = standardized * self.ptp + self.min
        return transformed

    def postprocess(self, transformed):
        standardized = (transformed - self.min) / self.ptp
        raw = standardized * self.data_ptp + self.data_min
        return raw


class GPR(torch.nn.Module):

    def __init__(self,
                 x_train,
                 y_train,
                 kernel,
                 noise,
                 lr_hyper=0.01,
                 lr_nn=0.001,
                 jitter=1e-6,
                 is_normalized=True):
        super().__init__()
        self.__free_noise = Parameter(unconstraint(noise))
        self.is_normalized = is_normalized
        if self.is_normalized:
            self.set_scalers(x_train, y_train)
        self.set_data(x_train, y_train)
        self.kernel = kernel
        self.initialize_optimizers(lr_hyper, lr_nn)
        self.jitter = jitter

    def initialize_optimizers(self, lr_hyper, lr_nn):
        hyper_params, nn_params = [], []
        for name, param in self.named_parameters():
            if "nn" in name:
                nn_params.append(param)
            else:
                hyper_params.append(param)
        self.opt_hyper = torch.optim.Adam(hyper_params, lr=lr_hyper)
        if nn_params:
            self.opt_nn = torch.optim.Adam(nn_params, lr=lr_nn)
        else:
            self.opt_nn = None

    def compute_common(self):
        K = self.kernel(self._x_train, self._x_train)
        K.diagonal().add_(self.noise)
        L = robust_cholesky(K, jitter=self.jitter)
        iK_y = torch.cholesky_solve(self._y_train, L, upper=False)
        return L, iK_y

    def compute_loss(self):
        L, iK_y = self.compute_common()
        quadratic = torch.sum(self._y_train * iK_y)
        logdet = L.diag().square().log().sum()
        constant = self.num_train * np.log(2 * np.pi)
        return 0.5 * (quadratic + logdet + constant)

    def optimize(self, num_iter, verbose=True, writer=None):
        self.train()
        pbar = range(num_iter)
        if verbose:
            pbar = tqdm(pbar)
        for i in pbar:
            self.opt_hyper.zero_grad()
            if self.opt_nn is not None:
                self.opt_nn.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.opt_hyper.step()
            if self.opt_nn is not None:
                self.opt_nn.step()
            if verbose:
                pbar.set_description(f"Iter: {i:02d} loss: {loss.item(): .2f}")
            if writer is not None:
                writer.add_scalar('loss', loss.item(), i)
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name, param.grad, i)
        self.eval()

    def forward(self, x_test, noise_free=False):
        # Pre-processing
        if self.is_normalized:
            x_test = self.x_scaler.preprocess(x_test)

        _x_test = torch.tensor(x_test, dtype=torch.float64)
        # Prediction
        with torch.no_grad():
            L, iK_y = self.compute_common()
            Ksn = self.kernel(_x_test, self._x_train)
            Kss_diag = self.kernel.diag(_x_test)
            iL_Kns = torch.linalg.solve_triangular(L, Ksn.t(), upper=False)
            _mean = Ksn @ iK_y
            var = Kss_diag - iL_Kns.square().sum(0).view(-1, 1)
            # TODO: variance might be zero when lengthscale is too large.
            if torch.any(var <= 0.0):
                print(var.ravel().numpy())
                raise ValueError("Predictive variance <= 0.0!")
            var.clamp_(min=self.jitter)
            if not noise_free:
                var += self.noise
            _std = var.sqrt()
        mean = _mean.numpy()
        std = _std.numpy()
        # Post-processing
        if self.is_normalized:
            mean = self.y_scaler.postprocess_mean(mean)
            std = self.y_scaler.postprocess_std(std)
        return mean, std

    def set_scalers(self, x_init, y_init):
        self.x_scaler = MinMaxScaler(x_init, expected_range=(-1.0, 1.0))
        self.y_scaler = StandardScaler(values=y_init)

    def set_data(self, x_train, y_train):
        if self.is_normalized:
            x_train = self.x_scaler.preprocess(x_train)
            y_train = self.y_scaler.preprocess(y_train)
        self._x_train = torch.tensor(x_train, dtype=torch.float64)
        self._y_train = torch.tensor(y_train, dtype=torch.float64)

    def add_data(self, x_new, y_new):
        if self.is_normalized:
            x_new = self.x_scaler.preprocess(x_new)
            y_new = self.y_scaler.preprocess(y_new)
        _x_new = torch.tensor(x_new, dtype=torch.float64)
        _y_new = torch.tensor(y_new, dtype=torch.float64)
        self._x_train = torch.vstack((self._x_train, _x_new))
        self._y_train = torch.vstack((self._y_train, _y_new))

    def get_data(self):
        x_train = self._x_train.numpy()
        y_train = self._y_train.numpy()
        if self.is_normalized:
            x_train = self.x_scaler.postprocess(x_train)
            y_train = self.y_scaler.postprocess_mean(y_train)
        return x_train, y_train

    @property
    def num_train(self) -> int:
        return self._x_train.size(0)

    @property
    def noise(self):
        return constraint(self.__free_noise)

    @noise.setter
    def noise(self, noise: float) -> None:
        with torch.no_grad():
            self.__free_noise.copy_(unconstraint(noise))


def rbf(dist, lengthscale):
    return torch.exp(-0.5 * torch.square(dist / lengthscale))


class TwoHiddenLayerTanhNN(torch.nn.Sequential):

    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(dim_input, dim_hidden))
        self.add_module("activation1", torch.nn.Tanh())
        self.add_module("linear2", torch.nn.Linear(dim_hidden, dim_hidden))
        self.add_module("activation2", torch.nn.Tanh())
        self.add_module("linear3", torch.nn.Linear(dim_hidden, dim_output))
        self.add_module("activation3", torch.nn.Softmax(dim=1))


class AK(torch.nn.Module):

    def __init__(self, amplitude, lengthscales, dim_input, dim_hidden,
                 dim_output):
        super().__init__()
        self.__free_amplitude = Parameter(unconstraint(amplitude))
        self.lengthscales = torch.tensor(lengthscales)
        np.set_printoptions(precision=2)
        print("Primitive lengthscales: ", self.lengthscales.numpy())
        self.nn = TwoHiddenLayerTanhNN(
            dim_input,
            dim_hidden,
            dim_output,
        ).double()

    def get_representations(self, x):
        z = self.nn(x)
        representations = z / z.norm(dim=1, keepdim=True)
        return representations

    def diag(self, x):
        return self.amplitude * torch.ones(x.size(0), 1, dtype=torch.float64)

    def forward(self, x_1, x_2):
        dist = torch.cdist(x_1, x_2)
        repre1 = self.get_representations(x_1)
        repre2 = self.get_representations(x_2)
        cov_mat = 0.0
        for i in range(self.num_lengthscales):
            attention_lengthscales = torch.outer(repre1[:, i], repre2[:, i])
            cov_mat += rbf(dist, self.lengthscales[i]) * attention_lengthscales
        attention_inputs = repre1 @ repre2.t()
        cov_mat *= self.amplitude * attention_inputs
        return cov_mat

    @property
    def num_lengthscales(self):
        return len(self.lengthscales)

    @property
    def amplitude(self):
        return constraint(self.__free_amplitude)

    @amplitude.setter
    def amplitude(self, amplitude):
        with torch.no_grad():
            self.__free_amplitude.copy_(unconstraint(amplitude))


def gaussian_entropy(std):
    entropy = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
    return entropy


class MyopicPlanning:

    def __init__(self, task_extent, rng, num_candidates, robot):
        self.task_extent = task_extent
        self.rng = rng
        self.num_candidates = num_candidates
        self.robot = robot

    def get(self, model, num_states=1):
        while len(self.robot.sampling_locations) == 0:
            # Propose candidate locations
            xs = self.rng.uniform(
                low=self.task_extent[0],
                high=self.task_extent[1],
                size=self.num_candidates,
            )
            ys = self.rng.uniform(
                low=self.task_extent[2],
                high=self.task_extent[3],
                size=self.num_candidates,
            )
            candidate_states = np.column_stack((xs, ys))
            # Evaluate candidates
            _, std = model(candidate_states)
            entropy = gaussian_entropy(std.ravel())
            diffs = candidate_states - self.robot.state[:2]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            # Normalized scores
            normed_entropy = (entropy - entropy.min()) / entropy.ptp()
            normed_dists = (dists - dists.min()) / dists.ptp()
            scores = normed_entropy - normed_dists
            # Append waypoint
            sorted_indices = np.argsort(scores)
            goal_states = candidate_states[sorted_indices[-num_states:]]
            self.robot.goal_states.append(goal_states.ravel())
            # Controling and sampling
            while self.robot.has_goal:
                self.robot.update(*self.robot.control())
        x_new = self.robot.commit_data()
        return x_new


class Logger:

    def __init__(self, eval_outputs):
        self.eval_outputs = eval_outputs
        self.means = []
        self.stds = []
        self.errors = []
        self.xs = []
        self.ys = []
        self.nums = []
        self.goals = []

    def append(self, mean, std, error, x, y, num):
        self.means.append(mean)
        self.stds.append(std)
        self.errors.append(error)
        self.xs.append(x)
        self.ys.append(y)
        self.nums.append(num)

    def to_numpy(self):
        self.means = np.hstack(self.means)
        self.stds = np.hstack(self.stds)
        self.errors = np.hstack(self.errors)
        self.xs = np.vstack(self.xs)
        self.ys = np.vstack(self.ys)
        self.nums = np.array(self.nums)


class Evaluator:

    def __init__(self, sensor, task_extent, eval_grid):
        self.task_extent = task_extent
        self.eval_grid = eval_grid
        self.setup_eval_inputs_and_outputs(sensor)
        self.log2pi = np.log(2 * np.pi)
        self.nums = []
        self.smses = []
        self.mslls = []
        self.nlpds = []
        self.rmses = []
        self.maes = []

    def setup_eval_inputs_and_outputs(self, sensor) -> None:
        xmin, xmax, ymin, ymax = self.task_extent
        num_x, num_y = self.eval_grid
        x = np.linspace(xmin, xmax, num_x)
        y = np.linspace(ymin, ymax, num_y)
        xx, yy = np.meshgrid(x, y)
        self.eval_inputs = np.column_stack((xx.ravel(), yy.ravel()))
        self.eval_outputs = sensor.sense(self.eval_inputs).reshape(-1, 1)

    def eval_prediction(self, model):
        _, y_train = model.get_data()
        mean, std = model(self.eval_inputs)
        error = np.fabs(mean - self.eval_outputs)
        self.nums.append(model.num_train)
        self.smses.append(self.calc_smse(error))
        self.mslls.append(self.calc_msll(error, std, y_train))
        self.nlpds.append(self.calc_nlpd(error, std))
        self.rmses.append(self.calc_rmse(error))
        self.maes.append(self.calc_mae(error))
        return mean, std, error

    def calc_log_loss(self, error, std):
        log_loss = 0.5 * self.log2pi + np.log(std) + 0.5 * np.square(
            error / std)
        return log_loss

    def calc_nlpd(self, error, std):
        nlpd = np.mean(self.calc_log_loss(error, std))
        return nlpd

    def calc_rmse(self, error):
        rmse = np.sqrt(np.mean(np.square(error)))
        return rmse

    def calc_mae(self, error):
        mae = np.mean(error)
        return mae

    def calc_smse(self, error):
        mse = np.mean(np.square(error))
        smse = mse / self.eval_outputs.var()
        return smse

    def calc_msll(self, error, std, y_train):
        log_loss = self.calc_log_loss(error, std)
        baseline = self.calc_log_loss(self.eval_outputs - y_train.mean(),
                                      y_train.std())
        msll = np.mean(log_loss - baseline)
        return msll


def print_metrics(model, evaluator):
    print(f"Data: {model.num_train:04d} | " +
          f"SMSE: {evaluator.smses[-1]:.4f} | " +
          f"MSLL: {evaluator.mslls[-1]:.4f} | " +
          f"NLPD: {evaluator.nlpds[-1]:.4f} | " +
          f"RMSE: {evaluator.rmses[-1]:.4f} | " +
          f"MAE : {evaluator.maes[-1]:.4f}")


def get_matplotlib_axes():
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    fig.subplots_adjust(
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
        hspace=0.2,
        wspace=0.2,
    )
    return axes


def plot_image(ax, values, title):
    matrix = values.reshape(eval_grid)
    ax.imshow(
        matrix,
        extent=env_extent if title == "Ground Truth" else task_extent,
    )
    workspace = plt.Rectangle(
        (task_extent[0], task_extent[2]),
        task_extent[1] - task_extent[0],
        task_extent[3] - task_extent[2],
        linewidth=3,
        edgecolor="white",
        alpha=0.8,
        fill=False,
    )
    ax.add_patch(workspace)
    ax.set_title(title)


def clear_axes(axes):
    for each in axes.ravel()[1:]:
        each.cla()


def set_limits(axes):
    for ax in axes.ravel()[1:]:
        ax.set_xlim(env_extent[:2])
        ax.set_ylim(env_extent[2:])


def pause():
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [exit(0) if event.key == "escape" else None],
    )
    plt.pause(1e-2)


###############################################################################

# Set random seed
rng = np.random.RandomState(seed)
torch.manual_seed(seed)
print(f"Set random seed to {seed} in random, numpy, and torch.")

# Get environment
env_name = "N17E073"
print(f"Preprocessing {env_name}.jpg...")
image = Image.open(f"./{env_name}.jpg").convert("L")
array = np.array(image).astype(np.float64)
env = transform.resize(array, (
    array.shape[0] // 10,
    array.shape[1] // 10,
))

# Get sensor
sensor = Sonar(rate=sensing_rate,
               env=env,
               env_extent=env_extent,
               noise_scale=noise_scale)

# Get initial data
x1 = rng.uniform(low=task_extent[0],
                 high=task_extent[1],
                 size=num_init_samples)
x2 = rng.uniform(low=task_extent[2],
                 high=task_extent[3],
                 size=num_init_samples)
x_init = np.column_stack((x1, x2))
y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)

robot = USV(
    init_state=np.array([x_init[-1, 0], x_init[-1, 1], np.pi / 2]),
    control_rate=control_rate,
    max_lin_vel=max_lin_vel,
    tolerance=tolerance,
    sampling_rate=sensing_rate,
)

kernel = AK(
    amplitude=init_amplitude,
    lengthscales=np.linspace(min_lengthscale, max_lengthscale, dim_output),
    dim_input=dim_input,
    dim_hidden=dim_hidden,
    dim_output=dim_output,
)

model = GPR(
    x_train=x_init,
    y_train=y_init,
    kernel=kernel,
    noise=init_noise,
    lr_hyper=lr_hyper,
    lr_nn=lr_nn,
    jitter=jitter,
)
model.optimize(num_iter=model.num_train, verbose=True)

evaluator = Evaluator(sensor=sensor,
                      task_extent=task_extent,
                      eval_grid=eval_grid)
logger = Logger(evaluator.eval_outputs)
mean, std, error = evaluator.eval_prediction(model)
logger.append(mean, std, error, x_init, y_init, model.num_train)

strategy = MyopicPlanning(
    task_extent=task_extent,
    rng=rng,
    num_candidates=num_candidates,
    robot=robot,
)

while model.num_train < max_num_samples:
    x_new = strategy.get(model=model)
    y_new = sensor.sense(x_new, rng).reshape(-1, 1)
    model.add_data(x_new, y_new)
    model.optimize(num_iter=len(y_new), verbose=False)
    mean, std, error = evaluator.eval_prediction(model)
    logger.append(mean, std, error, x_new, y_new, model.num_train)
    print_metrics(model, evaluator)

logger.to_numpy()
axes = get_matplotlib_axes()
plot_image(axes[0, 0], logger.eval_outputs, "Ground Truth")
for index, num in enumerate(logger.nums):
    clear_axes(axes)
    set_limits(axes)
    plot_image(axes[0, 1], logger.means[:, index], "Prediction")
    plot_image(axes[1, 0], logger.stds[:, index], "Uncertainty")
    plot_image(axes[1, 1], logger.errors[:, index], "Error")
    axes[1, 0].plot(logger.xs[:num_init_samples, 0],
                    logger.xs[:num_init_samples, 1],
                    '.',
                    color='k',
                    alpha=0.8)
    axes[1, 0].plot(logger.xs[num_init_samples:num, 0],
                    logger.xs[num_init_samples:num, 1],
                    '.-',
                    color='k',
                    alpha=0.8)
    pause()
