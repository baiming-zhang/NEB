"""Small CPU entry points for the paper data and reference weights."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Neural Eigenfluid Bases reproduction tools")
    checkout = Path.cwd() if (Path.cwd() / "data/paper").is_dir() else Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=checkout,
                        help="repository directory")
    sub = parser.add_subparsers(dest="command", required=True)
    plot = sub.add_parser("figures", help="plot the supplied paper measurements")
    plot.add_argument("--output", type=Path, default=Path("outputs/paper_figures"))
    weights = sub.add_parser("checkpoints", help="list and download reference checkpoints")
    weights.add_argument("--download", action="store_true")
    weights.add_argument("--only", nargs="*", help="checkpoint IDs to select")
    args = parser.parse_args()
    if args.command == "figures":
        from .figures import reproduce
        reproduce(args.root, args.output)
    else:
        from .checkpoints import run
        run(args.root, download=args.download, only=args.only)


if __name__ == "__main__":
    main()
