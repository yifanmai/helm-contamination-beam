"""Apache Beam specific code."""

from typing import Dict, List, Iterable

from apache_beam.utils.shared import Shared
import apache_beam as beam
from helm.benchmark.contamination.compute_contamination_metrics import load_scenarios_from_jsonl
from helm.benchmark.contamination.light_tokenizer import LightTokenizer

from contamination_fixes import create_ngram_index, create_all_contamination_stats, compute_scenario_document_contamination, get_tokenizer_for_normalization, AllContaminationStats


class ComputeContaminationStatsFn(beam.CombineFn):
    def __init__(
        self,
        scenario_data_path: str,
        n_values: List[int],
        normalization: str,
        shared_ngram_index: Shared,
    ) -> None:
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.tokenizer: LightTokenizer = get_tokenizer_for_normalization(normalization)
        self.shared_ngram_index = shared_ngram_index

    def setup(self, *args, **kwargs) -> None:
        self.scenarios = load_scenarios_from_jsonl(self.scenario_data_path)

        def init_shared_ngram_index():
            return create_ngram_index(self.scenarios, self.n_values, self.tokenizer)

        self.ngram_index = self.shared_ngram_index.acquire(init_shared_ngram_index)
        return super().setup(*args, **kwargs)

    def create_accumulator(self) -> AllContaminationStats:
        return create_all_contamination_stats(self.scenarios, self.n_values)

    def add_input(self, test_set_contamination_stats: AllContaminationStats, document: str) -> AllContaminationStats:
        return compute_scenario_document_contamination(
            test_set_contamination_stats,
            document,
            self.ngram_index,
            self.n_values,
            self.tokenizer,
        )

    def merge_accumulators(self, accumulators: Iterable[AllContaminationStats]) -> AllContaminationStats:
        assert accumulators
        merged_accumulator = accumulators[0]
        for accumulator in accumulators[1:]:
            for contamination_stats_key, contamination_stats in accumulator:
                merged_accumulator[contamination_stats_key].merge(contamination_stats)
        return merged_accumulator

    def extract_output(self, accumulator: AllContaminationStats) -> AllContaminationStats:
        return accumulator


def extract_summary_from_all_contamination_stats(all_contamination_stats: AllContaminationStats, tags: Dict[str, str]) -> str:
    return "\n".join(
        contamination_stats.generate_summary(tags) for contamination_stats in all_contamination_stats.values()
    )


class ComputeAndWriteContaminationStats(beam.PTransform):
    def __init__(self, scenario_data_path: str, n_values: List[int], normalization: str, tags: Dict[str, str], output_stats: str):
        self.scenario_data_path = scenario_data_path
        self.n_values = n_values
        self.normalization = normalization
        self.tags = tags
        self.output_stats = output_stats

    def expand(self, pcollection: beam.PCollection):
        shared_ngram_index = Shared()
        return (pcollection
            | "ComputeContaminationStats"
            >> beam.CombineGlobally(
                ComputeContaminationStatsFn(
                    scenario_data_path=self.scenario_data_path,
                    n_values=self.n_values,
                    normalization=self.normalization,
                    shared_ngram_index=shared_ngram_index,
                )
            )
            | "ExtractSummaryFromAllContaminationStats"
            >> beam.Map(extract_summary_from_all_contamination_stats, tags=self.tags)
            | "WriteSummaries" >> beam.io.WriteToText(self.output_stats)
        )
