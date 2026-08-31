# External CARLA Execution Runbook

## Scope

The portable bundles are produced and statically validated on the current Python
3.13 host. Physics execution is intentionally isolated to Ubuntu 22.04,
Python 3.10, CARLA 0.9.16, and Scenic 3.1.1. A dry-run result is not a
simulation result.

## Environment

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv
python3.10 -m venv .venv-carla
source .venv-carla/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-carla.txt
```

Install and unpack the CARLA 0.9.16 server separately, then start it:

```bash
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000
```

Confirm connectivity:

```bash
python scripts/carla_healthcheck.py --host 127.0.0.1 --port 2000
```

## Bundle Verification And Execution

```bash
cd /path/to/Paper_NewLocalLLM
cd artifacts/bundles/jurisdrive_71
sha256sum -c checksums.sha256
cd ../../..
scripts/run_external_carla.sh artifacts/bundles/jurisdrive_71
```

The runner enables synchronous mode, uses the contract's
`fixed_delta_seconds=0.05`, synchronizes Traffic Manager, and sets the fixed
seed. It restores world settings and destroys spawned actors in `finally`.

## Scenic Check

The generated `scenario.scenic` is a portable source artifact. Validate it in
the external environment before CARLA execution:

```bash
scenic artifacts/bundles/jurisdrive_71/scenario.scenic --simulate --count 1
```

The core reproducibility path is `ScenarioContractV1 -> Scenic 3.x -> CARLA
0.9.16`. ChatScene may be added as an optional frontend adapter but is not
required by this path.

## Recovery

- If health check times out, verify the CARLA process and RPC port.
- If spawning fails, retry the same contract and seed only after confirming the
  map and blueprint inventory. Do not silently change an observed attribute.
- If Scenic rejects the source, preserve the bundle and validation log; do not
  mark it executed.
- Run handcrafted 5 cases, stratified 20 cases, then the full 200 cases.
- Repeat each accepted contract twice and compare actor mapping, spawn points,
  event ordering, collision target, and core telemetry.
