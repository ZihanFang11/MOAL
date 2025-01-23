
import argparse
import sys


## Parameter setting
def parameter_parser():
    parser = argparse.ArgumentParser()

    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)

    parser.add_argument("--data_path", type=str, default="./data/", help="Path of datasets.")

    parser.add_argument("--save_file", type=str, default="./result/unseen=3_res.txt", help="Save experimental result.")
    parser.add_argument("--save_resp_path", type=str, default="./resp/", help="Save experimental result.")
    parser.add_argument("--config_name", type=str, default="./config/unseen3.yaml", help="Save experimental result.")

    parser.add_argument("--unseen_num", type=int, default=3)
    parser.add_argument("--unseen_label_index", type=int, default=-100)
    parser.add_argument("--fusion_type", type=str, default="attention", help="Fusion Methods: trust average weight attention")

    parser.add_argument("--device", default="0", type=str, required=False)
    parser.add_argument("--fix_seed", action='store_true', default=True, help="")
    parser.add_argument("--use_softmax", action='store_true', default=True, help="")
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 42.")
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument("--ratio", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--training_rate", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--valid_rate", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--weight_decay", type=float, default=0.15, help="Weight decay")

    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--knn', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--block', type=int, default=1, help='block') # for the example dataset, block can set 2 and more than 2
    parser.add_argument('--thre', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=1)
    return parser.parse_args()