# Human Gold Annotation Protocol

## 목적

seed `20260728`로 추출한 900건을 두 명의 사람 주석자가 원문을 직접 읽고
독립적으로 라벨링한다. 불일치와 판단 불가능 사례는 제3의 사람 주석자가
adjudication한다. 주석자에게 모델 출력, 기존 예측 라벨, sampling stratum은
제공하지 않는다.

## 라벨

- `car_to_car`: 서로 다른 두 도로 차량 사이의 접촉·충돌이 인정사실 또는
  법원의 판단으로 확인된다.
- `not_car_to_car`: 보행자, 자전거, 단독, 시설물, 동물 사고이거나 차량 간
  충돌 근거가 없다.
- 판단 불가능한 문서는 annotator 파일에서 임의 이진 라벨을 넣지 말고
  `uncertain`을 선택한 뒤 `notes`에 사유를 기록해 adjudication 대상으로 보낸다.

## 필수 필드

- `vehicle_count`: 사고에 참여한 서로 다른 도로 차량 수
- `collision_agent`: 충돌 행위를 한 차량의 원문 별칭
- `collision_target`: 충돌 대상 차량의 원문 별칭
- `legal_status`: `accepted_fact`, `party_claim`, `court_reasoning`, `unknown`
- `evidence_quotes`: 원문에서 그대로 복사한 최소 근거 span
- `notes`: 대명사, 동일차량, 다중충돌, 법적 지위 등 불일치 사유

## 절차

1. 각 주석자는 candidate ID와 source text만 포함된 blinded task 파일을 읽는다.
2. 차량 entity를 분리하고 `same_as` 표현을 확인한다.
3. 충돌 agent/target과 사건 순서를 표시한다.
4. 문장이 인정사실인지 당사자 주장인지 구분한다.
5. exact quote를 기록한 후 이진 사고 라벨을 결정한다.
6. Annotator A/B 파일을 독립적으로 완료한다.
7. Cohen's kappa가 0.80 미만이면 90건 pilot guideline을 수정하고 재라벨링한다.
8. 불일치와 `uncertain` 사례를 제3자가 검토해 `adjudicated.jsonl`을 완성한다.

## 기록 방식

표본과 빈 라벨 파일은 라이선스 자료가 있는 로컬 환경에서 생성한다. 이 디렉터리는
`artifacts/` 아래에 두며 Git에 올리지 않는다.

```bash
python -m jurisdrive sample-gold \
  --full-run-dir /path/to/full_run \
  --output-dir artifacts/gold_kit \
  --seed 20260728
```

사람 주석자에게는 `annotation_tasks_blinded.jsonl`만 전달한다. 내부 집계용
`annotation_tasks.jsonl`과 prediction 파일은 라벨링 종료 전까지 주석자에게
공개하지 않는다.

Annotator A와 B는 서로의 기록을 볼 수 없는 상태에서 각각
`annotator_a.jsonl`, `annotator_b.jsonl`을 오프라인으로 작성한다. 계정,
비밀번호, 세션 토큰, 원문, 주석 파일은 저장소에 커밋하지 않는다. 각 근거 quote는
`annotation_tasks_blinded.jsonl`의 원문에 정확히 포함되어야 한다. 라벨링이 끝날
때까지 모델 예측, sampling stratum, 상대 주석자의 라벨과 메모를 공개하지 않는다.

두 파일이 완료되면 입력 해시와 합의/검토 집합을 고정한다.

```bash
python scripts/freeze_dual_human_gold.py \
  --tasks artifacts/gold_kit/annotation_tasks.jsonl \
  --annotator-a /private/path/annotator_a.jsonl \
  --annotator-b /private/path/annotator_b.jsonl \
  --output-dir artifacts/gold_consensus
```

진행률은 다음 명령으로 확인한다.

```bash
python -m jurisdrive gold-status --gold-dir artifacts/gold_kit
```

## 평가 도구

라벨링 도중에는 상태와 입력 해시만 기록하고 정확도는 생성하지 않는다.

```bash
python scripts/evaluate_gold_set.py \
  --gold-dir artifacts/gold_kit \
  --output-dir artifacts/gold_evaluation
```

제3자 판정 900건이 모두 이진 라벨로 완성되면 다음이 생성된다.

- `metrics.json`: Cohen's kappa, coverage, selective risk, confusion matrix,
  precision, recall, F1, MCC, false-acceptance rate
- `metrics_table.csv`: 논문 표에 바로 넣을 수 있는 평면 테이블
- `disagreements.jsonl`: annotator 불일치 감사 목록
- `evaluation_manifest.json`: 입력/출력 SHA-256과 event-chain 검증 결과

표본이 sampling stratum별로 서로 다른 비율로 추출되었기 때문에, 라벨링 종료 후
집계 단계에서만 stratum 정보를 다시 결합해 단순 표본 지표와
inverse-probability population-weighted 지표를 함께 계산한다. 두 annotator 파일이
binary/`uncertain`으로 완료되고 adjudication의 900건이 모두 이진 라벨로 완성되기
전에는 F1, MCC 또는 kappa를 생성하지 않는다. `uncertain`은 이진 라벨로 강제
변환하지 않고 제3자 판정 대상으로 보존한다.
