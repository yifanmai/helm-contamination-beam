# Apache Beam implementation of HELM

## Installation

```
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3 venv

# Activate the virtual environment.
source venv/bin/activate

# Install requirements
pip install git+https://github.com/stanford-crfm/helm.git@4d704fc46ffe3c083878b81e2c5f60224170ca27
pip install apache-beam~=2.46.0
pip install bitarray~=2.7.3
```

## Running

Running without Apache Beam:
```bash
python3 run_contamination_local.py --input-data /path/to/input_data.jsonl --scenario-data /path/to/scenario_data.jsonl --input-format the_pile --output-stats /path/to/output_stats.jsonl
```

Running with Apache Beam:

```bash
python3 run_contamination_beam.py --input-data /path/to/input_data.jsonl --scenario-data /path/to/scenario_data.jsonl --input-format the_pile --output-stats /path/to/output_stats.jsonl
```

## API

Model developers should implement an Apache Beam pipeline that creates a `PCollection[str]` of documents, and then pass it to `ComputeAndWriteContaminationStats()` with the appropriate arguments.

Note: Each record in the `PCollection[str]` should contain an _entire_ document, not a single line from a document.

```python
with beam.Pipeline() as pipeline:
    _ = (
        pipeline
        # The model developer should modify these lines to read from the actual training set.
        | "Read" >> beam.io.ReadFromText(input_data)
        | "ExtractTextFromDocument" >> beam.Map(extract_text_from_document)
        # Call the HELM Contamination Apache Beam API.
        | "ComputeAndWriteContaminationStats" >> ComputeAndWriteContaminationStats(
            scenario_data_path=scenario_data,
            n_values=n_values,
            normalization=normalization,
            tags=tags
        )
    )
```
