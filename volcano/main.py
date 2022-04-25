from pathlib import Path
from urllib import request

import numpy as np
#  import rasterio
from scipy.io import savemat
#  from skimage import transform

import pypolo
from parse_arguments import parse_arguments


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


# preprocess_helens requires rasterio and scikit-image.
# rasterio might be tricky to install, so we've provided the preprocessed data
# and commented this function

#  def preprocess_helens(data_path):
#      name = "helens"
#      print(f"Preprocessing {name}...")
#      raster = rasterio.open(f"{data_path}/raw/10.2.1.1043901.dem")
#      array = raster.read(1).astype(np.float64)  # type: ignore
#      resized = transform.resize(array, output_shape=(129, 129))
#      resized = resized[15:-14, 15:-14]  # Exclude extreme values at the boundary
#      save_path = f"{data_path}/preprocessed/{name}.npy"
#      np.save(save_path, resized)
#      print(f"Saved to {save_path}.")


def get_environment(data_path):
    path = Path(f"{data_path}/preprocessed/helens.npy")
    # Uncomment the following block if you would like to download and process
    # the DEM file. Packages required for this step: rasterio, scikit-image
    #  if not path.is_file():
    #      Path(f"{data_path}/raw").mkdir(parents=True, exist_ok=True)
    #      Path(f"{data_path}/preprocessed").mkdir(parents=True, exist_ok=True)
    #      download_helens(data_path)
    #      preprocess_helens(data_path)
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


def get_data(args, rng, sensor, data_path):
    """Returns the training data."""
    x_path = Path(f"{data_path}/preprocessed/x_train.csv")
    y_path = Path(f"{data_path}/preprocessed/y_train.csv")
    if x_path.is_file():
        x_train = np.genfromtxt(x_path, delimiter=",")
    else:
        x1 = rng.uniform(args.task_extent[0],
                         args.task_extent[1],
                         size=args.max_num_samples)
        x2 = rng.uniform(args.task_extent[2],
                         args.task_extent[3],
                         size=args.max_num_samples)
        x_train = np.column_stack((x1, x2))
        np.savetxt(x_path, x_train, delimiter=",")
    if y_path.is_file():
        y_train = np.genfromtxt(y_path, delimiter=",")
    else:
        y_train = sensor.sense(x_train)
        np.savetxt(y_path, y_train, delimiter=",")
    return x_train, y_train.reshape(-1, 1)


def get_model(args, x_init, y_init):
    """Returns the chosen model."""
    kernel = pypolo.experiments.utilities.get_kernel(args)
    model = pypolo.models.GPR(
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
    data_path = "../data/volcano/"
    args = parse_arguments(verbose=True)
    rng = pypolo.experiments.utilities.seed_everything(args.seed)
    env = get_environment(data_path)
    sensor = get_sensor(args, env)
    x_train, y_train = get_data(args, rng, sensor, data_path)
    model = get_model(args, x_train, y_train)
    model.optimize(num_iter=args.num_train_iter, verbose=True)
    evaluator = get_evaluator(args, sensor)
    mean, std, error = evaluator.eval_prediction(model)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{args.output_dir}{args.kernel}.mat"
    savemat(
        save_path,
        {
            "mean": mean.reshape(args.eval_grid),
            "error": error.reshape(args.eval_grid),
            "std": std.reshape(args.eval_grid),
        },
    )
    print(f"Kernel: {args.kernel.upper()}")
    print(f"SMSE\t{evaluator.smses[0]:.4f}")
    print(f"MSLL\t{evaluator.mslls[0]:.4f}")
    print(f"NLPD\t{evaluator.nlpds[0]:.4f}")
    print(f"RMSE\t{evaluator.rmses[0]:.4f}")
    print(f"MAE\t{evaluator.maes[0]:.4f} ")
    print(f"Mean, error, and std matrices are saved to {save_path}")


if __name__ == "__main__":
    main()
