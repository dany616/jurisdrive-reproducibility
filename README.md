# JurisDrive

Reproducible code and compact evidence for **JurisDrive: Provenance-Bounded
Legal-to-Simulation Staging and Failure Localization**.

JurisDrive converts selected Korean traffic-accident judgments into
provenance-tagged candidate CARLA scenarios. The pipeline combines deterministic
three-way filtering, selective LLM resolution, evidence graphs, scenario
contracts, controlled simulation, and bounded assurance checks.

```text
N0 judgments -> N1 extraction -> N2 rule filter -> N3 selective Qwen
             -> N4 evidence graph -> N5 scenario contract/CARLA
             -> N6 telemetry and keyframe assurance
```

## Data source

- Original name: **민사법 LLM 사전학습 및 Instruction Tuning 데이터**
- English name: **Civil Law LLM Pretraining and Instruction Tuning Data**
- Source: [AI Hub dataset 71841](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%AF%BC%EC%82%AC%EB%B2%95%20LLM%20%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5%20%EB%B0%8F%20Instruction%20Tuning%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71841)

The licensed source dataset is not redistributed in this repository. Download
it from AI Hub and follow its access and use conditions. Raw judgments, model
weights, and large generated run directories remain outside Git.

## Quick verification

Python 3.11 or newer is recommended for the repository-level verification
environment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python scripts/verify_release.py
```

The command compiles the Python sources, runs the complete unit-test suite,
checks the frozen N0--N6 summary invariants, validates the CLI, and executes a
small synthetic graph-to-contract-to-dry-run example. It does not require the
licensed dataset, a model server, or CARLA.

## Reproduce with the licensed corpus

Place the prepared N0--N3 outputs in a `full_run` directory with this layout:

```text
full_run/
├── summary.json
├── output/car_to_car/*.json
└── ambiguous_done/
    ├── summary.json
    └── car_to_car/*.json
```

Run the corpus audit with explicit, machine-local paths:

```bash
python src/analysis/audit_current_data.py \
  --raw-dir /path/to/raw \
  --zeroshot-dir /path/to/zeroshot_done \
  --full-run-dir /path/to/full_run \
  --output-dir artifacts/audit
```

Then reproduce a deterministic stratified graph, contract, and dry-run batch:

```bash
python -m jurisdrive build-graph \
  --manifest artifacts/audit/final_car_to_car_manifest.jsonl \
  --full-run-dir /path/to/full_run \
  --tier-counts A=20,B=10,C=10 \
  --output-dir artifacts/graphs

python -m jurisdrive build-contract \
  --manifest artifacts/audit/final_car_to_car_manifest.jsonl \
  --full-run-dir /path/to/full_run \
  --tier-counts A=20,B=10,C=10 \
  --graph-dir artifacts/graphs \
  --output-dir artifacts/contracts

python -m jurisdrive compile \
  --manifest artifacts/audit/final_car_to_car_manifest.jsonl \
  --tier-counts A=20,B=10,C=10 \
  --graph-dir artifacts/graphs \
  --contract-dir artifacts/contracts \
  --output-dir artifacts/bundles
```

The generated manifest remains under the ignored `artifacts/` tree because it
contains judgment-derived per-record fields. Its source paths are relative to
`--full-run-dir`, so it can be regenerated and used without publishing local
machine paths or licensed material.

## Repository map

```text
jurisdrive/       core evidence, contract, simulation, and assurance package
src/              N-stage analysis and filtering snapshots
scripts/          audits, evaluations, and figure/table generators
tests/            deterministic unit and integration tests
configs/          portable examples and frozen experiment specifications
results/          compact aggregate summaries without per-record source data
examples/         synthetic data for a dataset-free smoke reproduction
```

## Reproducibility boundary

This checkout supports deterministic code verification, result reanalysis,
N0--N3 auditing when the licensed inputs are supplied, and N4--N5 dry-run
regeneration. CARLA, ChatScene, GPU model serving, and VLM experiments require
the separately documented external environments and should record their exact
versions, model identifiers, seeds, and endpoints.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the verification levels and
execution guidance.
