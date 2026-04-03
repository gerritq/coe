import argparse

from src.coe.coe_main import COEAnalyzer
from src.coe.coe_denoise import COEDenoiseAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["default", "denoise"], default="default")
    parser.add_argument(
        "--token_mode",
        type=str,
        default="last_token",
        choices=["last_token", "pooling", "horizontal"],
    )
    parser.add_argument("--diff_vectors", type=int, default=0)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--normalize", type=int, default=0)
    parser.add_argument("--save_viz", type=int, default=0)
    parser.add_argument("--classifier", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert args.smoke_test in (0, 1), "smoke_test must be 0 or 1"
    args.smoke_test = bool(args.smoke_test)
    assert args.diff_vectors in (0, 1), "diff_vectors must be 0 or 1"
    args.diff_vectors = bool(args.diff_vectors)
    assert args.prefix in (0, 1), "prefix must be 0 or 1"
    args.prefix = bool(args.prefix)
    assert args.normalize in (0, 1), "normalize must be 0 or 1"
    args.normalize = bool(args.normalize)
    assert args.save_viz in (0, 1), "save_viz must be 0 or 1"
    args.save_viz = bool(args.save_viz)
    assert args.classifier in (0, 1), "classifier must be 0 or 1"
    args.classifier = bool(args.classifier)

    print("=" * 50)
    print("Running with args:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)

    if args.mode == "denoise":
        analyzer = COEDenoiseAnalyzer(args=args)
    else:
        analyzer = COEAnalyzer(args=args)
    result = analyzer.run()
    print(result)


if __name__ == "__main__":
    main()
