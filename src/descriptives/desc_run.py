import argparse
import os
from argparse import Namespace

BASE_DIR = os.getenv("BASE_COE")
OUT_DIR = os.path.join(BASE_DIR, "output", "descriptives", "sandbox")
os.makedirs(OUT_DIR, exist_ok=True)

COMPARE_DATASETS = [
    "m4_wikihow_chatgpt",
    "m4_wikipedia_chatgpt",
    "m4_reddit_chatgpt",
    "m4_arxiv_chatgpt",
]


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="last_token")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--analysis", type=str, default="all")
    parser.add_argument("--dim", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    from src.descriptives.pca_analyzer import PCAAnalyzer
    from src.descriptives.pca_trajectory import PCATrajectoryAnalyzer
    from src.descriptives.sv_analyser import SVAnalyser
    from src.descriptives.topic_sv import TopicSVAnalyzer

    args = parse_args()
    if args.prefix not in (0, 1):
        raise ValueError("prefix must be 0 or 1")
    if args.smoke_test not in (0, 1):
        raise ValueError("smoke_test must be 0 or 1")
    args.prefix = bool(args.prefix)
    args.smoke_test = bool(args.smoke_test)

    if args.analysis in ["ld", "all"]:
        analyzer = PCAAnalyzer(model_name=args.model)
        result = analyzer.run(args)
        print(result)

    if args.analysis in ["traj", "all"]:
        traj_analyzer = PCATrajectoryAnalyzer(model_name=args.model)
        traj_result = traj_analyzer.run(args)
        print(traj_result)

    if args.analysis in ["sv", "all"]:
        sv_analyzer = SVAnalyser(model_name=args.model)
        sv_result = sv_analyzer.run(args)
        print(sv_result)

    if args.analysis in ["topic_sv", "all"]:
        topic_sv_analyzer = TopicSVAnalyzer(model_name=args.model)
        topic_sv_result = topic_sv_analyzer.run(args)
        print(topic_sv_result)


if __name__ == "__main__":
    main()
