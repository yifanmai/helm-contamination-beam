import json
from typing import Callable, Dict, List

import apache_beam as beam

from contamination_parser import create_parser, parse_tags_from_args
from contamination_beam import ComputeAndWriteContaminationStats


def extract_text_from_the_pile_document(document: str) -> str:
    return json.loads(document)["text"]


def extract_text_from_raw_document(document: str) -> str:
    return document


def main():
    parser = create_parser()
    args = parser.parse_args()

    tags: Dict[str, str] = parse_tags_from_args(args)
    n_values: List[int] = [5, 9, 13]  # TODO: Pick the N values
    extract_text_from_document: Callable[[str], str]
    
    if args.input_format == "raw":
        extract_text_from_document = extract_text_from_raw_document
    elif args.input_format == "the_pile":
        extract_text_from_document = extract_text_from_the_pile_document
    else:
        raise NotImplementedError(f"Unknown input format {args.input_format}")

    # The model developer should pass in the appropriate PipelineOptions here.
    with beam.Pipeline() as pipeline:
        _ = (
            pipeline
            # The model developer should modify these lines to read from the actual training set.
            | "Read" >> beam.io.ReadFromText(args.input_data)
            | "ExtractTextFromDocument" >> beam.Map(extract_text_from_document)
            # Call the HELM Contamination Apache Beam API.
            | "ComputeAndWriteContaminationStats" >> ComputeAndWriteContaminationStats(
                scenario_data_path=args.scenario_data,
                n_values=n_values,
                normalization=args.normalization,
                tags=tags,
                output_stats=args.output_stats,
            )
        )
    print(f"Wrote results to {args.output_stats}")
    


if __name__ == "__main__":
    main()
