import json
from typing import Callable, Dict, List

from apache_beam.utils.shared import Shared
import apache_beam as beam

from helm.benchmark.contamination.compute_contamination_metrics import (
    load_scenarios_from_jsonl,
)
from helm.benchmark.contamination.light_tokenizer import (
    LightTokenizer,
    DefaultTokenizer,
)

from .contamination_parser import create_parser, parse_tags_from_args
from .contamination_fixes import create_ngram_index, create_test_set_contamination_stats, compute_scenario_document_contamination


class ComputeContaminationMetricsFn(beam.CombineFn):
    def __init__(
        self,
        scenario_data_path: str,
        n_values: List[int],
        normalization: str,
        shared_ngram_index: Shared,
    ):
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.tokenizer: LightTokenizer
        if normalization == "none":
            self.tokenizer = LightTokenizer()
        elif normalization == "default":
            self.tokenizer = DefaultTokenizer()
        else:
            raise ValueError("Unknown normalization type")
        self.shared_ngram_index = shared_ngram_index

    def setup(self, *args, **kwargs):
        print(f"Loading scenario data from {self.scenario_data_path}")
        self.scenarios = load_scenarios_from_jsonl(self.scenario_data_path)

        def init_shared_ngram_index():
            return create_ngram_index(self.scenarios, self.n_values, self.tokenizer)

        self.ngram_index = self.shared_ngram_index.acquire(init_shared_ngram_index)
        return super().setup(*args, **kwargs)

    def create_accumulator(self):
        return create_test_set_contamination_stats(self.scenarios, self.n_values)

    def add_input(self, test_set_contamination_stats, document):
        return compute_scenario_document_contamination(
            test_set_contamination_stats,
            document,
            self.ngram_index,
            self.n_values,
            self.tokenizer,
        )

    def merge_accumulators(self, accumulators):
        assert accumulators
        merged_accumulator = accumulators[0]
        for accumulator in accumulators[1:]:
            for contamination_stats_key, contamination_stats in accumulator:
                merged_accumulator[contamination_stats_key].merge(contamination_stats)
        return merged_accumulator

    def extract_output(self, accumulator):
        return accumulator


def extract_text_from_the_pile_document(document: str):
    return json.loads(document)["text"]


def extract_text_from_raw_document(document: str):
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

    shared_ngram_index = Shared()
    with beam.Pipeline() as pipeline:
        _ = (
            pipeline
            | "Read" >> beam.io.ReadFromText(args.input_data)
            | "ExtractDocumentText" >> beam.Map(extract_text_from_document)
            | "CountGlobally"
            >> beam.CombineGlobally(
                ComputeContaminationMetricsFn(
                    scenario_data_path=args.scenario_data,
                    n_values=n_values,
                    normalization=args.normalization,
                    shared_ngram_index=shared_ngram_index,
                )
            )
            | "GetSummaries"
            >> beam.Map(
                lambda test_set_contamination_stats: "\n".join(
                    contamination_stats.generate_summary()
                    for contamination_stats in test_set_contamination_stats.values()
                )
            )
            | "Print" >> beam.Map(print)
        )


main()
