# VII. Experimental Evaluation

## A. Research Questions and Experimental Protocol

We evaluated JurisDrive with four research questions. **RQ1** asks whether the
legal-discourse-aware cascade reduces the number of judgments that require LLM
inference while preserving an explicit unresolved state. **RQ2** asks whether
the selected cases can be compiled into evidence-grounded graphs and scenario
contracts without losing the original textual span. **RQ3** measures the
reliability and scalability of asynchronous vLLM inference. **RQ4** examines
whether executable contracts yield deterministic CARLA traces and whether the
multimodal assurance stage accepts the resulting telemetry and keyframes.

The source corpus contains 76,291 Korean judgment records. All N0--N3 counts
were taken from the frozen pipeline archive; the archive was not overwritten or
re-inferred. The N4--N5 static experiment used a fixed 400-case set comprising
Tier A/B/C cases, while the new serving experiment used the first 400 cases in
the naturally ordered ambiguous-route manifest. A separate, fixed 16-case
workload was used for the concurrency sweep so that every concurrency setting
processed the same requests. The server was warmed before timed measurements.

The workstation contained two Intel Xeon Gold 6530 processors (64 physical
cores and 128 logical processors in total), 512 GB of installed host memory
(251.379 GiB visible to the benchmark under WSL), and two NVIDIA RTX PRO 4500
Blackwell GPUs with 32,623 MiB of memory per device.
The experiments ran under Ubuntu 24.04.4 LTS on WSL2. CARLA 0.9.13 was executed
with Python 3.12.3, and `Qwen/Qwen3.5-35B-A3B-FP8` was served as `qwen35-vlm`
from the vLLM 0.23.0 container with CUDA 13.0.2. The serving configuration used
tensor parallelism 2, a 32,768-token context limit, and `MAX_NUM_SEQS=8`. All
second-stage requests used temperature 0, thinking disabled, and a constrained
JSON schema. GPU and CPU load were sampled once per second during the earlier
400-request classifier run and every 5 s during the 200-run N5--N6 benchmark.

For historical comparison, we retained the original supercomputer logs for
`qwen35-27b`. Those logs include 200-request batches at client concurrency 16,
32, and 64 and the complete 2,524-request ambiguous-case run at concurrency 16.
Because the archived and workstation runs use different model deployments and
hardware, their absolute times are reported separately rather than treated as a
controlled hardware comparison.

## B. Corpus Reduction and Selective Routing

Table I summarizes the frozen selection cascade. The deterministic rules
resolved 73,767 of 76,291 records without an LLM call and routed only 2,524
records (3.308%) to Qwen. The second stage recovered 431 additional car-to-car
cases, rejected 1,357 cases, and retained 736 cases as unresolved. The final
partition satisfies the conservation invariant
2,902 + 72,653 + 736 = 76,291.

**TABLE I. FROZEN N0--N3 CASCADE COUNTS**

| Stage or branch | Records | Corpus share (%) |
|---|---:|---:|
| Structured judgments | 76,291 | 100.000 |
| Rule: Accept | 2,471 | 3.239 |
| Rule: Reject | 71,296 | 93.453 |
| Routed to Qwen | 2,524 | 3.308 |
| Qwen: Accept | 431 | 0.565 |
| Qwen: Reject | 1,357 | 1.779 |
| Qwen: unresolved | 736 | 0.965 |
| Final Accept | 2,902 | 3.804 |

The rule stage processed the full corpus in 61.32 s. The archived asynchronous
LLM stage processed 2,524 routed cases in 2,688.29 s with zero request failures,
corresponding to 0.939 cases/s. Thus, the measured hybrid runtime for these two
stages was 45.83 min. If every record had incurred the archived mean batch cost
of 1.0651 s, an all-LLM pass would require approximately 22.57 h. Under this
linear-throughput assumption, selective routing reduces the projected wall time
by 29.55 times and avoids 96.692% of LLM calls. The 22.57-h figure is an
extrapolation, not a directly measured all-LLM run.

## C. Evidence Grounding and Contract Validity

The 400-case static experiment completed without a graph-generation crash or a
schema failure. All 400 evidence spans were exact substrings of their source
texts, and all 400 Scenario Contracts passed schema validation. Contract gating
assigned 104 cases (26.00%) to `needs_defaults`, 189 (47.25%) to
`needs_review`, and 107 (26.75%) to `blocked`. None of the 100 Tier-C cases was
automatically promoted to an executable state. A separate 200-case dry run
preserved the strict `not_executed` state and produced valid bundle checksums in
all cases.

**TABLE II. STATIC GROUNDING AND COMPILATION CHECKS**

| Check | n | Passed | Rate (%) |
|---|---:|---:|---:|
| Evidence Graph schema and crash-free generation | 400 | 400 | 100.0 |
| Exact source-span containment | 400 | 400 | 100.0 |
| Scenario Contract schema | 400 | 400 | 100.0 |
| Tier-C auto-promotion prevented | 100 | 100 | 100.0 |
| Dry-run strict `not_executed` | 200 | 200 | 100.0 |
| Dry-run checksum validity | 200 | 200 | 100.0 |

Exact-span containment is a provenance-integrity result: it verifies that a
quoted span exists verbatim in the judgment. It does not by itself establish
that the extracted relation is semantically correct; that claim is reserved for
the human gold evaluation described in Sec. VII-F.

## D. Asynchronous vLLM Scalability

Table III reports the matched 16-request workstation sweep. Increasing
concurrency from 1 to 8 reduced wall time from 43.22 to 11.52 s and increased
throughput from 0.370 to 1.389 requests/s, a 3.75-times speedup. Concurrency 16
produced only a small additional throughput gain (1.448 requests/s) while mean
latency increased from 5.08 to 7.91 s. This plateau is consistent with the
server-side limit of eight active sequences: additional client threads mainly
queue requests instead of expanding GPU batching. We therefore used
concurrency 8 for the 400-request reliability run.

**TABLE III. WORKSTATION ASYNCHRONOUS SCALING (FIXED 16-CASE WORKLOAD)**

| Concurrency | Wall time (s) | Throughput (req/s) | Mean latency (s) | P95 latency (s) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 43.223 | 0.370 | 2.664 | 3.425 | 1.00x |
| 2 | 25.356 | 0.631 | 3.120 | 4.043 | 1.71x |
| 4 | 15.656 | 1.022 | 3.637 | 4.738 | 2.76x |
| 8 | 11.522 | 1.389 | 5.078 | 7.371 | 3.75x |
| 16 | 11.054 | 1.448 | 7.911 | 10.462 | 3.91x |

The archived supercomputer measurements show the same qualitative saturation.
For 200 requests, increasing concurrency from 16 to 32 reduced wall time from
133.31 to 92.68 s (1.44x), whereas concurrency 64 required 92.46 s and provided
no material improvement. Together, the two environments indicate that client
parallelism should be increased only until the serving engine's batching and
memory limits are reached.

The full workstation run processed 400/400 requests successfully in 274.89 s
(4.58 min), or 1.455 requests/s. Mean, median, and P95 request latency were
5.422, 5.354, and 7.080 s, respectively. The output contained 79 car-to-car,
171 not-car-to-car, and 150 unresolved decisions. Mean GPU utilization across
both devices was 61.42%, P95 utilization was 90%, and peak utilization was 96%.
Mean and peak allocated GPU memory were 88.43% and 92.66% of device capacity.
No CUDA out-of-memory, fatal NCCL, engine, or request error occurred. The low
mean host-CPU load (3.11%) confirms that the measured stage was GPU-bound.

**TABLE IV. HARDWARE LOAD AND RUNTIME FOR THE 400-REQUEST RUN**

| Workload | Success | Wall time (s) | Throughput (req/s) | GPU util. mean/P95 (%) | GPU memory mean/peak (%) | CPU util. mean/peak (%) |
|---|---:|---:|---:|---:|---:|---:|
| Async ambiguous-case classification, c=8 | 400/400 | 274.892 | 1.455 | 61.42 / 90.00 | 88.43 / 92.66 | 3.11 / 5.00 |

For the same 400 filenames, the workstation decisions agreed with the archived
deployment on 366 cases (91.5%), with Cohen's kappa = 0.866. This is a
cross-deployment consistency result, not accuracy against human ground truth.

## E. Runtime Performance and Hardware Load

We measured the bounded N5--N6 execution path on six accepted executable
contracts (IDs 25, 71, 99, 367, 460, and 692). The scenarios were balanced at
33--34 repetitions, giving 200 measured CARLA runs and 200 N6 evaluations.
CARLA used 12 scenario-isolated server processes, synchronous stepping at
0.05 s, low-quality off-screen rendering, 800 x 450 RGB capture, and 80 or 120
frames according to the contract. One unmeasured warm-up preceded each
server/map stream. All 200 physical runs completed, satisfied the simulation
and hard-collision constraints, and produced the required collision. The
measured CARLA phase required 91.78 min, corresponding to 2.179 runs/min; mean
and P95 per-run latency were 224.74 and 343.27 s, respectively.

The same 200 run artifacts were then evaluated by the N6 VLM using at most
three collision-centered keyframes. Six scenario-specific warm-ups were
excluded. Thirty round-robin requests were issued sequentially to estimate
single-request latency, while the remaining 170 requests were issued
asynchronously at concurrency eight to measure throughput. Sequential mean/P95
latency was 1.93/2.36 s. The asynchronous phase achieved 1.320 requests/s and
5,402.6 total tokens/s, with mean/P95 request latency of 6.01/7.42 s. All
200/200 responses passed the telemetry-and-keyframe contract, and no response
was flagged for repair or manual review.

**TABLE V. 200-RUN CARLA AND N6 VLM RUNTIME BENCHMARK**

| Measure | Result |
|---|---:|
| Unique executable scenarios | 6 |
| Measured CARLA runs / N6 VLM evaluations | 200 / 200 |
| Simulation and hard-constraint pass | 200 / 200 |
| CARLA batch wall-clock (12 workers) | 91.78 min |
| CARLA run latency, mean / P95 | 224.74 / 343.27 s |
| CARLA throughput | 2.179 runs/min |
| N6 VLM pass / manual-review flags | 200/200 / 0 |
| Sequential VLM latency, mean / P95 (`n=30`) | 1.93 / 2.36 s |
| Asynchronous VLM latency, mean / P95 (`c=8`, `n=170`) | 6.01 / 7.42 s |
| Asynchronous VLM throughput (`c=8`) | 1.320 req/s |
| Asynchronous VLM token throughput (`c=8`) | 5,402.6 tokens/s |

The two stages exhibited complementary resource profiles. Across 1,166 CARLA
resource samples, mean/P95/maximum host CPU utilization was
19.45/21.15/28.19%. The aggregate CARLA process load averaged 2,303.84% when
normalized to one logical processor, i.e., approximately 23.04 logical-core
equivalents, and reached 29.15 equivalents at its maximum. CARLA resident-set
memory averaged 37.78 GiB and reached 45.70 GiB, whereas total host memory use
averaged 54.50 GiB (21.68% of the WSL-visible allocation), with P95 and maximum
values of 56.79 GiB (22.59%) and 57.34 GiB (22.81%). GPU compute utilization
during this low-quality off-screen CARLA phase remained below 0.03% on average,
while mean device-memory use was 2.23 GiB on GPU 0 and 1.70 GiB on GPU 1.

In contrast, the measured VLM phase was GPU-bound. Across 36 samples, mean GPU
utilization was 66.19% on GPU 0 and 68.75% on GPU 1; both devices reached 96%
at P95, with maxima of 99% and 97%, respectively. Mean allocated memory was
28.67 GiB (89.99%) on GPU 0 and 28.26 GiB (88.71%) on GPU 1, and mean device
power was 113.25 and 107.94 W. Host CPU utilization remained at 2.95% on
average (3.21% at P95), and total host memory averaged 18.87 GiB (7.51% of the
WSL-visible allocation). These measurements show that the 12-worker simulator
is primarily CPU- and host-memory-oriented, whereas N6 throughput is governed
by GPU compute and memory occupancy.

Server startup and warm-up were excluded from measured latency; the cached VLM
service restart alone required 293.43 s. Consequently, Table V characterizes
steady-state batch throughput rather than cold-start latency. It covers N5
CARLA execution and N6 multimodal assurance only and must not be described as a
200-case end-to-end rerun of N0--N6.

## F. Human Gold Evaluation and Accuracy Reporting

To measure semantic accuracy, we prepared a 900-case stratified gold set with a
fixed seed (20260728): 200 rule-positive, 200 rule-negative, 150 Qwen-positive,
150 Qwen-negative, and 200 unresolved cases. Two annotators independently label
accident class, vehicle count, collision agent and target, legal-discourse
status, and exact evidence spans. They may select `uncertain` rather than force
an unsupported binary label. A third reviewer adjudicates disagreements and all
uncertain cases. The annotation interface hides pipeline predictions from the
two primary reviewers, verifies every quoted span against the source, performs
atomic writes, and records a hash-chained audit event for every save.

After adjudication, the evaluation tool reports raw sample and
inverse-probability population-weighted Precision, Recall, F1, MCC,
false-acceptance rate, coverage, selective risk, and Cohen's kappa. Abstentions
are excluded from covered-set confusion metrics and are instead reflected in
coverage and selective risk. The blinded Qwen-only comparator was precomputed
for all 900 tasks in 543.36 s at concurrency 8 with a 512-token output limit:
319 car-to-car, 484 not-car-to-car, and 97 abstentions were recorded, with one
automatic retry and zero final failures. These are predictions awaiting human
scoring, not accuracy results. At the time of this report, the 900 tasks,
comparator outputs, and all recording/evaluation tools are ready, but the human
labels are not complete.
Consequently, no human-ground-truth classifier accuracy is claimed in this
version. The tool deliberately withholds `metrics.json` until all adjudicated
labels are complete.

**TABLE VI. GOLD-SET COMPOSITION AND CURRENT STATUS**

| Stratum | Population | Gold sample | Current human labels |
|---|---:|---:|---:|
| Rule ACCEPT | 2,471 | 200 | Pending |
| Rule REJECT | 71,296 | 200 | Pending |
| Qwen ACCEPT | 431 | 150 | Pending |
| Qwen REJECT | 1,357 | 150 | Pending |
| Unresolved | 736 | 200 | Pending |
| **Total** | **76,291** | **900** | **0/900 adjudicated** |

The three comparison files are ready: rule-only covers 400/900 cases, the
hybrid covers 700/900, and Qwen-only covers 803/900 while abstaining on 97.
Coverage here is an operational property only; selective risk and accuracy
remain unavailable until human adjudication is complete.

## G. Threats to Validity

First, the concurrency sweep contains 16 fixed cases; it is designed to measure
serving behavior, not semantic performance. Second, the archived supercomputer
and current workstation results use different deployments, so only within-host
speedups are causal comparisons. Third, exact-span and schema checks do not
replace human semantic annotation. Fourth, the CARLA evidence consists of six
bounded collision scenarios and may not represent the complete distribution of
maps, weather, non-collision outcomes, and long-horizon interactions. Finally,
the VLM and the scenario-generation stack share related model priors; human
review remains necessary to estimate false acceptance. These limitations are
made explicit by separating measured, estimated, and pending values throughout
the tables.
