"""Argparse related code, which should be merged upstream."""

import argparse
from typing import Dict


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data", type=str, required=True, help="Path to your training data"
    )
    parser.add_argument(
        "--scenario-data",
        type=str,
        required=True,
        help="Path to scenario data (benchmarking data)",
    )
    parser.add_argument(
        "--output-stats", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        required=True,
        help="The format of your input file for your training data, e.g. raw, custom, the_pile",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Other tags, such as whether the input data is for pretraining or instruction tuning. \
            Format: key value pairs seperated by semicolons. E.g. --tags k1=v1;k2=v2",
        default=None,
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="default",
        help="What normalization and tokenization strategy to apply",
    )
    # parser.add_argument(
    #     "--output-ngrams",
    #     type=str,
    #     default=None,
    #     help="Path to the file of overlapped ngrams. If not given, ngrams will not be output.",
    # )
    return parser


def parse_tags_from_args(args: Dict) -> Dict:
    tags = {}
    if args.tags is not None:
        try:
            for tag_pair in args.tags.split(";"):
                k, v = tag_pair.split("=")
                tags[k] = v
        except ValueError:
            raise ValueError("Failed to parse the tags.")
    return tags
