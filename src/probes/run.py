import argparse
from argparse import Namespace

from src.probes.probe_logistic import LogisticProbeBase
from src.probes.probe_logistic_m import LogisticManifoldProbeBase


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["logistic", "logistic_m"],
        default="logistic",
    )

    parser.add_argument("--token_mode", type=str, default="last_token")
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--ood", type=str, default="")
    parser.add_argument("--normalize_scores", type=int, default=0)
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--ablation_set", type=str, choices=["human", "machine", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.normalize_scores not in (0, 1):
        raise ValueError("normalize_scores must be 0 or 1")

    if args.ood.strip():
        args.ood = args.ood.split(" ")
    else:
        args.ood = []

    args.prefix = bool(args.prefix)
    args.smoke_test = bool(args.smoke_test)
    args.normalize_scores = bool(args.normalize_scores)
    args.manifold = args.mode == "logistic_m"

    if args.mode == "logistic_m":
        analyzer = LogisticManifoldProbeBase(model_name=args.model)
    else:
        analyzer = LogisticProbeBase(model_name=args.model)
    result = analyzer.run(args)
    print(result)


if __name__ == "__main__":
    main()
