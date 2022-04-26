import configargparse


def parse_arguments(verbose=True):
    parser = configargparse.ArgParser()
    parser.add_argument("-c",
                        "--config",
                        required=False,
                        is_config_file=True,
                        help="Config file path.")
    parser.add_argument("--output-dir",
                        type=str,
                        default="./outputs",
                        help="Directory for outputs.")
    parser.add_argument("--figure-dir",
                        type=str,
                        default="./figures",
                        help="Directory for figures.")
    # Experiment Settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--env-name",
                        type=str,
                        required=True,
                        help="Environment name.")
    parser.add_argument("--kernel",
                        type=str,
                        required=True,
                        help="Kernel name.")
    parser.add_argument("--sensing-rate",
                        type=float,
                        help="Sensing data update rate.")
    parser.add_argument("--env-extent",
                        type=float,
                        action="append",
                        help="Extent of the environment.")
    parser.add_argument("--task-extent",
                        type=float,
                        action="append",
                        help="Extent of the sampling task.")
    parser.add_argument("--eval-grid",
                        type=int,
                        action="append",
                        help="Evaluation grid size [num_x, num_y].")
    parser.add_argument("--noise-scale",
                        type=float,
                        default=1,
                        help="Observational noise scale.")
    parser.add_argument("--num-train",
                        type=int,
                        default=600,
                        help="Number of training samples.")
    parser.add_argument("--num-iter",
                        type=int,
                        default=1000,
                        help="Number of training iterations.")
    # Kernel Settings
    parser.add_argument("--init-amplitude",
                        type=float,
                        default=1.0,
                        help="Initial amplitude hyper-parameter.")
    parser.add_argument("--init-lengthscale",
                        type=float,
                        default=0.5,
                        help="Initial amplitude hyper-parameter.")
    parser.add_argument("--init-noise",
                        type=float,
                        default=1.0,
                        help="Initial noise variance hyper-parameter.")
    parser.add_argument("--lr-hyper",
                        type=float,
                        default=0.01,
                        help="Learning rate for hyper-parameters.")
    parser.add_argument("--lr-nn",
                        type=float,
                        default=0.001,
                        help="Learning rate for neural-network parameters.")
    parser.add_argument("--jitter",
                        type=float,
                        default=1e-6,
                        help="Small positive value for numerical stability.")
    parser.add_argument("--dim-input",
                        type=int,
                        default=2,
                        help="Number of input dimensions of neural networks")
    parser.add_argument("--dim-hidden",
                        type=int,
                        default=10,
                        help="Number of hidden dimensions of neural networks")
    parser.add_argument("--dim-output",
                        type=int,
                        default=10,
                        help="Number of output dimensions of neural networks")
    parser.add_argument("--min-lengthscale",
                        type=float,
                        default=0.01,
                        help="Min primitive lengthscale of AK")
    parser.add_argument("--max-lengthscale",
                        type=float,
                        default=0.5,
                        help="Max primitive lengthscale of AK")
    parser.add("--ak-enable-softmax",
               dest="ak_use_softmax",
               action="store_true")
    parser.add("--ak-disable-softmax",
               dest="ak_use_softmax",
               action="store_false")
    parser.set_defaults(ak_use_softmax=True)
    args = parser.parse_args()
    if verbose:
        print(parser.format_values())
    return args
