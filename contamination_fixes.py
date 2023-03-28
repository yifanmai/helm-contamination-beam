"""Contamination related code, which should be merged upstream."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any

from nltk import ngrams

from helm.benchmark.contamination.contamination_stats import (
    ContaminationStats,
    PART_INPUT,
    PART_REF,
)
from helm.benchmark.contamination.light_scenario import LightScenario
from helm.benchmark.contamination.light_tokenizer import (
    DefaultTokenizer,
    LightTokenizer,
)
from helm.common.hierarchical_logger import hlog


@dataclass(frozen=True)
class FakeScenarioSpec:
    """Placeholder because the LightScenario JSON does not have a real ScenarioSpec."""

    # TODO: LightScenario JSON should use real ScenarioSpecs.
    args: Dict[str, Any]

    def __hash__(self):
        return hash(tuple((k, self.args[k]) for k in sorted(self.args.keys())))


@dataclass(frozen=True)
class EntryContaminationKey:
    """Unique key representing a string entry in an instance in a scenario.

    A 'string entry' refers to either the input or a single reference of a single instance."""

    # TODO: This should be a real ScenarioSpec
    scenario_spec: FakeScenarioSpec
    instance_id: int
    part: str
    """Either PART_INPUT or PART_REF"""


@dataclass(frozen=True)
class ContaminationStatsKey:
    """Key for the dict"""

    # TODO: This should be a real ScenarioSpec
    scenario_spec: FakeScenarioSpec
    n: int


@dataclass(frozen=True)
class NgramIndex:
    ngram_to_entry_contamination_keys: Dict[Tuple[str, ...], Set[EntryContaminationKey]]
    """Dict of n-grams to the key of every entry in the test set that contains that n-gram."""


AllContaminationStats = Dict[ContaminationStatsKey, ContaminationStats]


def create_ngram_index(
    scenarios: List[LightScenario], n_values: List[int], tokenizer: LightTokenizer
) -> NgramIndex:
    """Given a list of scenarios and n values, initialize the stats and ngram_index data structures"""
    ngram_index: Dict[Tuple[str, ...], Set[EntryContaminationKey]] = {}
    for scenario in scenarios:
        scenario_spec = FakeScenarioSpec(scenario.scenario_spec)
        hlog(f"Building ngram indexes for {str(scenario_spec)}")
        for n in n_values:
            for i in range(len(scenario.light_instances)):
                instance = scenario.light_instances[i]
                # compute input ngrams
                input_unigrams = tokenizer.tokenize(instance.input)
                for input_ngram in ngrams(input_unigrams, n):
                    if input_ngram not in ngram_index:
                        ngram_index[input_ngram] = set()
                    ngram_index[input_ngram].add(
                        EntryContaminationKey(scenario_spec, i, PART_INPUT)
                    )

                # compute reference ngrams
                for reference in instance.references:
                    reference_unigrams = tokenizer.tokenize(reference)
                    for reference_ngram in ngrams(reference_unigrams, n):
                        if reference_ngram not in ngram_index:
                            ngram_index[reference_ngram] = set()
                        ngram_index[reference_ngram].add(
                            EntryContaminationKey(scenario_spec, i, PART_REF)
                        )
    return NgramIndex(ngram_index)


def create_all_contamination_stats(
    scenarios: List[LightScenario], n_values: List[int]
) -> AllContaminationStats:
    all_contamination_stats: AllContaminationStats = {}
    for scenario in scenarios:
        scenario_spec = FakeScenarioSpec(scenario.scenario_spec)
        for n in n_values:
            # Initizlize a stats instance for every pair of <scenario, n>
            stats: ContaminationStats = ContaminationStats.from_scenario(
                scenario, stats_tags={"N": n}
            )
            all_contamination_stats[
                ContaminationStatsKey(scenario_spec, n)
            ] = stats
    return all_contamination_stats


def compute_scenario_document_contamination(
    all_contamination_stats: AllContaminationStats,
    document: str,
    ngram_index: Dict[Tuple[str, ...], List[EntryContaminationKey]],
    n_values: List[int],
    tokenizer: LightTokenizer,
) -> AllContaminationStats:
    """Given a document, compute a contamination stats for each n and each scenario"""
    document_unigrams = tokenizer.tokenize(document)
    for n in n_values:
        for document_ngram in ngrams(document_unigrams, n):
            entry_contamination_keys = (
                ngram_index.ngram_to_entry_contamination_keys.get(document_ngram)
            )
            if not entry_contamination_keys:
                continue
            for entry_contamination_key in entry_contamination_keys:
                all_contamination_stats[
                    ContaminationStatsKey(entry_contamination_key.scenario_spec, n)
                ].write_dirty(
                    entry_contamination_key.instance_id, entry_contamination_key.part
                )
    return all_contamination_stats


def get_tokenizer_for_normalization(normalization: str):
    if normalization == "none":
        return LightTokenizer()
    elif normalization == "default":
        return DefaultTokenizer()
    else:
        raise ValueError("Unknown normalization type")
