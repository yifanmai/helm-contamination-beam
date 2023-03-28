from typing import Dict, List

from contamination_parser import create_parser, parse_tags_from_args
from contamination_fixes import create_ngram_index, create_all_contamination_stats, compute_scenario_document_contamination, get_tokenizer_for_normalization

from helm.benchmark.contamination.compute_contamination_metrics import load_scenarios_from_jsonl
from helm.benchmark.contamination.document_reading_processor import DocumentReadingProcessor


def main():
    parser = create_parser()
    args = parser.parse_args()

    tags: Dict[str, str] = parse_tags_from_args(args)

    n_values: List[int] = [5, 9, 13]  # TODO: Pick the N values
    scenarios = load_scenarios_from_jsonl(args.scenario_data)
    tokenizer = get_tokenizer_for_normalization(args.normalization)
    ngram_index = create_ngram_index(scenarios, n_values, tokenizer)
    all_contamination_stats = create_all_contamination_stats(scenarios, n_values)
    document_generator = DocumentReadingProcessor(
        file_path=args.input_data, file_format=args.input_format
    ).get_document_generator()
    for document in document_generator:
        compute_scenario_document_contamination(
            all_contamination_stats,
            document,
            ngram_index,
            n_values,
            tokenizer,
        )
    
    summaries = "\n".join(contamination_stats.generate_summary(tags) for contamination_stats in all_contamination_stats.values())
    with open(args.output_stats, "w") as f:
        f.write(summaries)
    print(f"Wrote results to {args.output_stats}")
    

if __name__ == "__main__":
    main()
