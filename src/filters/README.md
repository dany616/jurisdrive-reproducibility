# Verified Filter Snapshot

이 디렉터리는 다음 경로에서 2026-07-28에 복사한 N-stage 필터 구현 스냅샷이다.

```text
<original-workspace>/LocalLLM/zeroshot_test/pipelines/car_to_car_filter
```

대규모 결과와 HPC 실행 helper는 복사하지 않았다. Rule/Qwen 배치를 다시 실행할 때는
기존 `LocalLLM` 실행 경로와 vLLM resolver를 사용하고, 이 디렉터리는 논문 구현
고정본과 코드 검토에 사용한다.

## Source Hashes

| File | SHA-256 |
|---|---|
| `__init__.py` | `47e803b654760acacbddd1c2bae08b4e5e17b03f47eea82ca6c806546c1845f2` |
| `llm_filter.py` | `2d5f36eb03fd4487a7fd8acd6b79a2f5893e4168ac133c594b5be5903e198f5d` |
| `pipeline.py` | `2771eac9abc645cf698e1c18f7431f7ef6427a76497f2eba369c969d604d2a93` |
| `prompt_templates.py` | `1404306760b865f45f0955c15ab7a7631bff626aa254679633372f58517e9ce3` |
| `rule_filter.py` | `4cbeb5ddf4eebd7a51b4a5d21867a44d272fa6fdbe69c22477881c1146cc70df` |
| `run_ambiguous_llm_filter.py` | `07b4bd65a6dbeadb69f54d38532a9aa35db408f8fca224ffa778dad284c25cf0` |

`run_ambiguous_llm_filter.py`는 원본 프로젝트의 `Structured_output_test.py`와
SLURM/vLLM resolver에 의존한다. 해당 의존성을 이 연구 저장소에 복제하지 않는
이유는 cluster별 실행 설정과 논문 분석 산출물을 분리하기 위해서다.
