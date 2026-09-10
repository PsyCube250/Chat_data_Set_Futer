"""Command-line entry point for the chat dataset purification pipeline."""
import argparse
import os

import tool01_preprocess
import tool02_classify
import tool03_label
import tool04_integrate


def cmd_preprocess(args):
    tool01_preprocess.run(args.input, args.output)


def cmd_classify(args):
    tool02_classify.run(args.input, args.output_dir)


def cmd_label(args):
    tool03_label.run(args.candidates, args.output)


def cmd_integrate(args):
    tool04_integrate.run(
        args.stats,
        args.labels,
        args.classified,
        args.output,
    )


def cmd_all(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    preprocessed = os.path.join(output_dir, "preprocessed.json")

    print("=== Tool 01: Preprocess ===")
    tool01_preprocess.run(args.input, preprocessed)

    print("\n=== Tool 02: Classify ===")
    tool02_classify.run(preprocessed, output_dir)

    print(
        "\nTool 03 is interactive. Run it separately after reviewing "
        "candidates.json."
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Chat dataset purification pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("preprocess", help="Tool 01: clean, redact and summarize")
    p1.add_argument("--input", required=True)
    p1.add_argument("--output", required=True)
    p1.set_defaults(func=cmd_preprocess)

    p2 = sub.add_parser("classify", help="Tool 02: classify target utterances")
    p2.add_argument("--input", required=True)
    p2.add_argument("--output-dir", required=True)
    p2.set_defaults(func=cmd_classify)

    p3 = sub.add_parser("label", help="Tool 03: interactive human labeling")
    p3.add_argument("--candidates", required=True)
    p3.add_argument("--output", required=True)
    p3.set_defaults(func=cmd_label)

    p4 = sub.add_parser("integrate", help="Tool 04: calibrate and build dataset")
    p4.add_argument("--stats", required=True)
    p4.add_argument("--labels", required=True)
    p4.add_argument("--classified", required=True)
    p4.add_argument("--output", required=True)
    p4.set_defaults(func=cmd_integrate)

    pall = sub.add_parser(
        "all",
        help="Run Tool 01 + Tool 02; Tool 03 remains interactive",
    )
    pall.add_argument("--input", required=True)
    pall.add_argument("--output-dir", required=True)
    pall.set_defaults(func=cmd_all)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
