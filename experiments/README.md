# RQ3/RQ4 CARLA Fidelity and Assurance Experiments

This directory preserves the frozen RQ3/RQ4 design and the tooling used to plan,
materialize, and summarize those experiments. The current manuscript reports a
completed 24-case/96-run prepared CARLA study and a completed sanitized Task-A
evaluation. The large execution artifacts are not redistributed here. The
commands below define the protocol for additional versioned experiment runs.

## Frozen denominators

- 24 unique judgment cases: four topologies x six cases.
- 12 rule-routed and 12 Qwen-routed cases; each topology has three of each.
- Two fixed seeds per case and two repeats per seed: 96 clean fidelity runs.
- Six faults per case: 144 faulty artifacts.
- 24 clean controls + 144 faults: 168 base assurance artifacts.
- 72 mutable (`inferred/defaulted`) and 72 immutable/evidence-conflict faults.
- Five evaluation methods over the same 168 artifacts: 840 method evaluations.
- A guarded repair may change at most one field per iteration and runs for at
  most three repair iterations.

The existing six-scenario, 200-run benchmark is a separate repeated runtime
experiment and must not be merged with these denominators.

## 1. Select and freeze the 24 cases

Edit `configs/carla_assurance_24_preregistration.json`. Fill every candidate ID,
matching `jurisdrive_<candidate_id>` scenario ID, contract path, and the manually
confirmed topology. The Qwen route corresponds to the historical manifest's
`source_stage=llm`. Do not set `selection_frozen=true` until all 24 judgments
have been checked; record the UTC freeze time in `frozen_at_utc`.

Validate without writing artifacts:

```powershell
python scripts/prepare_carla_assurance_experiments.py validate `
  --config configs/carla_assurance_24_preregistration.json
```

Create a visibly non-executable draft while selection is pending:

```powershell
python scripts/prepare_carla_assurance_experiments.py plan `
  --config configs/carla_assurance_24_preregistration.json `
  --output-dir artifacts/migration_runs/<timestamp>/rq3_rq4_plan_draft `
  --allow-pending
```

After the selection is frozen, omit `--allow-pending`. The tool refuses to
overwrite an existing output directory and records the config checksum.

## 2. Record the 96 clean fidelity runs

The generated `fidelity_schedule.jsonl` fixes seed and repeat assignments.
Only an actual CARLA 0.9.13 run may change `execution_status` to `completed`.
Populate the predefined compile, launch, actor-target, topology, event-order,
TTC, impact-speed, hard-constraint, telemetry-hash, fallback, and failure fields.

Do not replace a failed case after the selection freeze. Record its failure
reason. A row with a missing field remains unassessed for that metric.

## 3. Materialize fault copies

Materialization copies an oracle contract/result into a new directory; it never
edits the clean bundle. Example:

```powershell
python scripts/prepare_carla_assurance_experiments.py inject `
  --source-bundle <clean-bundle> `
  --output-dir <new-fault-bundle> `
  --fault-type speed_pose_perturbation `
  --variant speed
```

Fault types are `actor_target_swap`, `required_collision_omission`,
`event_order_violation`, `speed_pose_perturbation`, `map_lane_mismatch`, and
`mismatched_keyframes`. The last type also requires `--donor-bundle` from a
different scenario.

Mutable contract faults are labeled `awaiting_carla_rerun` and
`injection_verified=false`. They enter no metric denominator until an actual
rerun confirms the intended fault. An event-order fault is blocked when a
contract has fewer than two grounded events; the tool does not invent one.

## 4. Summarize measured records

```powershell
python scripts/prepare_carla_assurance_experiments.py summarize-fidelity `
  --records <completed-fidelity.jsonl> `
  --output-dir <new-fidelity-summary-dir>

python scripts/prepare_carla_assurance_experiments.py summarize-assurance `
  --records <completed-assurance.jsonl> `
  --output-dir <new-assurance-summary-dir>
```

Summaries exclude unexecuted and unverified rows, preserve explicit
denominators, report replay agreement separately from telemetry-hash identity,
and write both JSON and UTF-8 CSV. Assurance output includes confusion metrics,
false acceptance/rejection, repair, regression, immutable-edit guard, manual
review, fault-type, and method strata.

## Verification

```powershell
python -m unittest tests.test_experiments -v
python -m unittest discover -s tests -v
python scripts/prepare_carla_assurance_experiments.py --help
```
