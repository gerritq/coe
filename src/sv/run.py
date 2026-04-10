import argparse
from argparse import Namespace

from src.sv.sv_denoise import DenoiseSVBase
from src.sv.sv_denoise_layer import DenoiseLayerSVBase
from src.sv.sv_ldp import LDPSVBase
from src.sv.sv_ldp_by_layer import LDPByLayerSVBase
from src.sv.sv_main import SVBase


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    # steering strategy mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "denoise", "denoise_layer", "ldp", "ldp_by_layer"],
        default="default",
    )

    # hidden-state extraction mode
    parser.add_argument("--token_mode", type=str, default="last_token")

    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--ood", type=str, default="")
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--normalize_scores", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    if args.normalize_scores not in (0, 1):
        raise ValueError("normalize_scores must be 0 or 1")
    if args.token_mode != "last_token":
        raise ValueError("This script expects --token_mode last_token.")

    if args.ood.strip():
        args.ood = args.ood.split(" ")
    else:
        args.ood = []

    args.prefix = bool(args.prefix)
    args.smoke_test = bool(args.smoke_test)
    args.normalize_scores = bool(args.normalize_scores)
    args.manifold = args.mode in ("denoise", "denoise_layer", "ldp", "ldp_by_layer")

    if args.mode == "denoise":
        analyzer = DenoiseSVBase(model_name=args.model)
    elif args.mode == "denoise_layer":
        analyzer = DenoiseLayerSVBase(model_name=args.model)
    elif args.mode == "ldp":
        analyzer = LDPSVBase(model_name=args.model)
    elif args.mode == "ldp_by_layer":
        analyzer = LDPByLayerSVBase(model_name=args.model)
    else:
        analyzer = SVBase(model_name=args.model)

    result = analyzer.run(args)
    print(result)


if __name__ == "__main__":
    main()
