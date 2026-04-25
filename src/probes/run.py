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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.token_mode not in {"last_token", "pooling"}:
        raise ValueError("token_mode must be one of: last_token, pooling")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.ood not in (0, 1):
        raise ValueError("ood must be 0 or 1")

    args.smoke_test = bool(args.smoke_test)
    args.ood = bool(args.ood)
    args.model_name = args.model

    analyzer = LinearProbing(args=args)
    analyzer.run(args)


if __name__ == "__main__":
    main()
