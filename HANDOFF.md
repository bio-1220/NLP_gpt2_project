# Experiment Handoff

이 문서는 데이터/코드 담당자가 GPU 학습 담당자에게 넘기는 실행 규격이다.
세부 연구 배경은 `proposal.md`에 두고, 여기서는 바로 학습과 평가에 필요한
파일, 필드, 우선순위만 정리한다.

## 1. Goal

현재 핵심 질문은 다음과 같다.

```text
감정 정보가 GPT-2 계열 모델의 시 continuation 성능에 도움이 되는가?
```

초기 실험에서는 global emotion prefix와 emotion SFT가 CHRF를 뚜렷하게
개선하지 못했다. 따라서 추가 실험은 새 주제로 갈아타기보다, 실패 원인을
분해하는 진단 실험으로 둔다.

## 2. Data Setup

외부 데이터는 git에 올리지 않는다. 새 환경에서는 아래 명령으로 다시 만든다.

```powershell
py -3 scripts\prepare_external_data.py
```

기본 출력:

```text
data/processed/emotion_en_{train,dev,test}.jsonl
data/processed/emotion_ko_{train,dev,test}.jsonl
data/processed/poem_ko_{train,dev,test}.jsonl
```

간단한 schema check:

```powershell
py -3 scripts\smoke_test_data_bridge.py
```

## 3. Common Schemas

### Emotion Classification

공통 label set:

```text
0 sadness
1 joy
2 love
3 anger
4 fear
5 surprise
```

주요 필드:

```text
text
label
label_id
```

### Poem Continuation

주요 필드:

```text
prefix    -> 첫 3줄
target    -> 나머지 줄
full_text -> 전체 시
```

조건부 generation 파일에는 다음 필드가 추가된다.

```text
model_input           -> test-time generation prompt
conditioned_full_text -> LM fine-tuning text
target                -> reference continuation
```

## 4. Recommended Run Priority

시간이 부족하면 아래 순서대로만 돌린다.

| Priority | Experiment | Files |
|---|---|---|
| 1 | Korean baseline | `poem_ko_{split}.jsonl` |
| 2 | KPoEM line oracle trajectory | `poem_ko_{split}_line_traj.jsonl` |
| 3 | KPoEM line average trajectory | `poem_ko_{split}_line_traj_avg.jsonl` |
| 4 | KPoEM line random trajectory | `poem_ko_{split}_line_traj_random.jsonl` |
| 5 | Shakespeare Method A oracle | `sonnet_en_{split}_sent_oracle_cont.jsonl` |
| 6 | Shakespeare Method B average | `sonnet_en_{split}_sent_avg_cont.jsonl` |

KPoEM line trajectory가 가장 중요하다. 이 실험은 global emotion label이 너무
coarse해서 실패했는지, line-level 구조화된 감정 흐름은 도움이 되는지 확인한다.

## 5. Diagnostic Data Builders

### KPoEM Oracle Emotion

```powershell
py -3 scripts\prepare_kpoem_oracle_emotion.py
```

출력:

```text
poem_ko_{train,dev,test}_oracle.jsonl
emotion_kpoem_prefix_{train,dev,test}.jsonl
kpoem_oracle_emotion_summary.json
```

용도:

```text
emotion_kpoem_prefix_* -> KOTE-trained classifier가 KPoEM 감정을 맞히는지 확인
poem_ko_*_oracle      -> oracle global emotion prefix generation
```

### KPoEM Line-Level Emotion Trajectory

```powershell
py -3 scripts\prepare_kpoem_line_trajectory.py
```

출력:

```text
poem_ko_{train,dev,test}_line_traj.jsonl
poem_ko_{train,dev,test}_line_traj_avg.jsonl
poem_ko_{train,dev,test}_line_traj_random.jsonl
kpoem_line_trajectory_summary.json
```

사용법:

```text
fine-tuning text: conditioned_full_text
test-time prompt: model_input
reference: target
```

### Shakespeare Sentiment Trajectory Method A/B

```powershell
py -3 scripts\prepare_shakespeare_sentiment_trajectory.py --continuous
```

출력:

```text
sonnet_en_{train,dev,heldout}_sent_oracle_cont.jsonl
sonnet_en_{train,dev,heldout}_sent_avg_cont.jsonl
shakespeare_sentiment_trajectory_summary_cont.json
```

의미:

```text
Method A oracle -> 각 sonnet의 gold 14-line sentiment trajectory를 prefix로 사용
Method B average -> train 평균 sentiment trajectory를 모든 sonnet에 공통 prefix로 사용
```

주의:

```text
heldout split은 target이 비어 있다.
dev split은 TRUE_sonnets_held_out_dev.txt에서 target을 가진다.
default discrete 평균은 거의 0으로 반올림되므로 `_cont` 사용을 우선 추천한다.
Method C predicted planner는 별도 모델과 leakage 관리가 필요하므로 이번 handoff 범위 밖이다.
```

## 6. Evaluation

기본 metric은 CHRF다.

비교는 같은 decoding setting에서 진행한다.

```text
baseline vs oracle trajectory
oracle trajectory vs average trajectory
oracle trajectory vs random trajectory
```

가능하면 per-poem CHRF도 저장한다. 평균 점수만 있으면 작은 데이터셋에서
해석이 흔들릴 수 있다.

## 7. Storyline

발표에서는 모든 실험을 다 나열하기보다 아래 흐름으로 묶는다.

```text
1. Emotion supervision이 poem continuation에 도움이 될 것이라고 가정했다.
2. Global emotion prefix와 sequential SFT는 CHRF를 개선하지 못했다.
3. 실패 원인을 보기 위해 더 구조화된 line-level trajectory를 실험한다.
4. Oracle/average/random 비교로 "감정 내용"과 "prefix 형식"을 분리한다.
5. 결론은 감정 분류 능력이 곧바로 생성 제어 능력으로 이어지지는 않는다는 것이다.
```

