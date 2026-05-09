import argparse
from argparse import Namespace

from src.probes.probe_main import LinearProbing


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--token_mode", type=str, default="last_token")
    parser.add_argument("--ood", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--training_size", type=int, default=None)
    parser.add_argument("--folder", type=str, default="sandbox")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.token_mode not in {"last_token", "pooling"}:
        raise ValueError("token_mode must be one of: last_token, pooling")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")
    if args.mode not in {"default", "pca", "meta", "meta_attn", "meta_no_pca", "mlp"}:
        raise ValueError("mode must be one of: default, pca, meta, meta_attn, meta_no_pca, mlp")

    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)
    args.model_name = args.model

    analyzer = LinearProbing(args=args)
    analyzer.run(args)


if __name__ == "__main__":
    main()
