# Training Handoff

이 문서는 데이터/전처리 담당 단계에서 학습 담당자에게 넘기는 handoff 문서입니다. 현재 레포는 **외부 데이터 준비와 모델-비의존 glue layer**까지 준비된 상태입니다. 실제 학습/평가 스크립트는 아직 작성해야 합니다.

## 1. 현재 완료된 것

### 데이터 전처리

다음 외부 데이터셋을 다운로드하고 `data/processed/` 아래 JSONL로 변환하는 스크립트를 추가했습니다.

```text
scripts/prepare_external_data.py
```

처리 대상:

| 용도 | 데이터셋 | 출력 |
|---|---|---|
| 영어 감정 분류 | `dair-ai/emotion` | `emotion_en_{train,dev,test}.jsonl` |
| 한국어 감정 분류 | KOTE strict 6-class subset | `emotion_ko_{train,dev,test}.jsonl` |
| 한국어 시 생성 | KPoEM line-level 재구성 | `poem_ko_{train,dev,test}.jsonl` |

외부 raw/processed 데이터는 repo에 커밋하지 않습니다. `.gitignore`에 포함되어 있습니다.

### 실험 glue utilities

모델 checkpoint 형식과 무관하게 쓸 수 있는 공통 유틸을 추가했습니다.

```text
experiments/data_bridge.py
experiments/prompts.py
experiments/metrics.py
scripts/smoke_test_data_bridge.py
```

이 유틸들은 아직 학습을 수행하지 않습니다. 학습 코드가 읽을 데이터 포맷, prompt 포맷, 평가 포맷을 고정하는 역할입니다.

## 2. 데이터 재생성 방법

처음 clone한 환경에서는 `data/processed/`가 없으므로 아래 명령을 실행해야 합니다.

```powershell
py -3 scripts\prepare_external_data.py
```

생성 결과 예시:

```text
data/processed/emotion_en_train.jsonl
data/processed/emotion_en_dev.jsonl
data/processed/emotion_en_test.jsonl

data/processed/emotion_ko_train.jsonl
data/processed/emotion_ko_dev.jsonl
data/processed/emotion_ko_test.jsonl

data/processed/poem_ko_train.jsonl
data/processed/poem_ko_dev.jsonl
data/processed/poem_ko_test.jsonl
```

데이터 bridge가 정상 동작하는지 확인:

```powershell
py -3 scripts\smoke_test_data_bridge.py
```

현재 smoke test에서 확인된 count:

```text
emotion_en_train: 16000
emotion_ko_train: 11880
poem_ko_train: 368
CHRF identity check: 100.0
```

## 3. Processed Dataset Schema

### 3.1 Emotion Classification JSONL

파일:

```text
data/processed/emotion_en_{train,dev,test}.jsonl
data/processed/emotion_ko_{train,dev,test}.jsonl
```

공통 label set:

```text
0 sadness
1 joy
2 love
3 anger
4 fear
5 surprise
```

예시:

```json
{
  "id": "48132",
  "text": "ㅋ 난 삼겹살먹고, 들어가는중~~ 좀있다봐",
  "label": "joy",
  "label_id": 1,
  "source_labels": ["기쁨", "즐거움/신남"],
  "target_label_counts": {"joy": 10, "love": 2},
  "target_rater_votes": {"joy": 4},
  "prompt_label_ko": "기쁨"
}
```

학습에는 보통 `text`, `label_id`만 쓰면 됩니다. 나머지는 분석/검증용 metadata입니다.

### 3.2 Korean Poem Continuation JSONL

파일:

```text
data/processed/poem_ko_{train,dev,test}.jsonl
```

예시:

```json
{
  "id": "kpoem-5",
  "poem_id": 5,
  "title": "또 다른 고향",
  "poet": "윤동주",
  "prefix": "고향에 돌아온 날 밤에\n내 백골(白骨)이 따라와 한방에 누웠다.\n어둔 방은 우주로 통하고",
  "target": "하늘에선가 소리처럼 바람이 불어 온다.\n...",
  "full_text": "고향에 돌아온 날 밤에\n...",
  "num_lines": 17,
  "emotion_metadata": {}
}
```

학습 방식은 두 가지 중 하나를 선택할 수 있습니다.

```text
1. full_text 전체 LM fine-tuning
2. prefix 이후 target token에만 loss를 주는 continuation fine-tuning
```

처음 baseline은 `full_text` 전체 LM fine-tuning이 가장 단순합니다. 과제의 continuation setting에 더 정확히 맞추려면 `prefix` 부분 loss masking을 추가하면 됩니다.

## 4. 전처리 정책 요약

### 4.1 영어 emotion

`dair-ai/emotion`을 그대로 사용합니다.

```text
sadness, joy, love, anger, fear, surprise
```

### 4.2 한국어 emotion

KOTE는 원래 43-class multi-label dataset입니다. 영어 데이터셋과 맞추기 위해 strict 6-class single-label subset으로 변환했습니다.

최종 mapping:

| Target | KOTE labels |
|---|---|
| sadness | 슬픔, 서러움, 절망 |
| joy | 기쁨, 행복, 즐거움/신남 |
| love | 아껴주는, 환영/호의 |
| anger | 화남/분노, 짜증, 증오/혐오 |
| fear | 공포/무서움, 불안/걱정 |
| surprise | 놀람, 경악, 당황/난처 |

Strict filtering:

```text
1. rater별 label을 6-class target으로 변환
2. 한 rater 안에서 여러 target class로 갈라지면 그 rater vote 제외
3. 최소 3명 이상이 같은 target class에 동의한 sample만 keep
4. 최다 class 동률이면 drop
```

주의: KOTE는 온라인 댓글 데이터라 텍스트 도메인이 시와 다릅니다. 이 점은 report limitation에 써야 합니다.

### 4.3 한국어 poem

KPoEM poem-level 파일은 줄바꿈이 사라진 문단 형태라 첫 3행 continuation task에 바로 쓸 수 없습니다. 따라서 line-level 파일을 받아 `poem_id` 기준으로 행을 재구성했습니다.

```text
prefix = first 3 lines
target = remaining lines
```

KPoEM emotion labels는 main training에는 쓰지 않습니다. `emotion_metadata`로만 보존합니다.

## 5. Glue Utility 사용법

### 5.1 Emotion dataset loader

```python
from experiments.data_bridge import EmotionJsonlDataset

dataset = EmotionJsonlDataset("data/processed/emotion_ko_train.jsonl")
example = dataset[0]

text = example["text"]
label_id = example["label_id"]
```

Tokenizer가 있을 경우:

```python
batch = dataset.collate_fn([dataset[0], dataset[1]], tokenizer=tokenizer, max_length=128)

input_ids = batch["token_ids"]
attention_mask = batch["attention_mask"]
labels = batch["labels"]
```

### 5.2 Poem continuation loader

```python
from experiments.data_bridge import PoemContinuationJsonlDataset

dataset = PoemContinuationJsonlDataset("data/processed/poem_ko_train.jsonl")
example = dataset[0]

prefix = example["prefix"]
target = example["target"]
full_text = example["full_text"]
```

### 5.3 Emotion prefix prompt

```python
from experiments.prompts import build_emotion_prompt

prompt = build_emotion_prompt(prefix, "sadness", language="ko")
```

출력:

```text
[감정: 슬픔]
<첫 3행>
```

영어:

```python
prompt = build_emotion_prompt(prefix, "sadness", language="en")
```

출력:

```text
[emotion: sadness]
<first 3 lines>
```

### 5.4 CHRF metric

```python
from experiments.metrics import chrf_corpus_score

score = chrf_corpus_score(hypotheses, references)
```

`sacrebleu`가 설치되어 있으면 정식 CHRF를 사용합니다. 없으면 fallback character n-gram F-score를 사용합니다. 최종 report 실험에서는 가능하면 `sacrebleu`를 설치하고 정식 CHRF를 쓰는 것을 권장합니다.

## 6. 다음에 구현해야 할 것

현재 아직 없는 학습/평가 스크립트:

```text
train_emotion.py
train_poem.py
evaluate_generation.py
checkpoint adapter
```

추천 구현 순서:

1. `train_emotion.py`
   - 영어: GPT-2 backbone + 6-class classifier
   - 한국어: KoGPT2 backbone + 6-class classifier
   - class imbalance가 있으므로 class weight 고려

2. `train_poem.py`
   - 영어: 기존 sonnet generation baseline과 연결
   - 한국어: `skt/kogpt2-base-v2` + KPoEM continuation
   - baseline은 poem SFT only

3. `evaluate_generation.py`
   - generated continuation 저장
   - reference target과 CHRF 계산
   - output JSONL 저장

4. Emotion prefix experiment
   - emotion classifier로 prefix의 emotion 예측
   - `[emotion: label]` 또는 `[감정: label]`을 붙여 generation
   - random emotion prefix control도 같이 구현

5. Sequential SFT experiment
   - emotion classifier fine-tuned checkpoint에서 shared backbone만 가져오기
   - generation LM head로 poem SFT 진행

## 7. Checkpoint 담당자에게 필요한 정보

학습 담당자가 영어 GPT-2 checkpoint를 넘겨받을 때 반드시 확인해야 할 것:

```text
1. checkpoint 파일 경로
2. checkpoint 형식
   - 우리 GPT2Model state_dict인지
   - HuggingFace GPT2LMHeadModel인지
   - classifier head까지 포함되어 있는지
3. tokenizer
   - gpt2 그대로인지
   - custom tokenizer인지
4. model size
   - hidden size
   - number of layers
   - number of heads
5. checkpoint가 어떤 task로 학습된 것인지
   - pretrain only
   - sentiment/emotion fine-tuned
   - sonnet generation fine-tuned
```

이 정보가 없으면 adapter를 확정하기 어렵습니다.

## 8. 실험 Matrix

최종적으로 목표하는 실험:

| ID | Language | Setup | Purpose |
|---|---|---|---|
| E0 | English | Sonnet SFT only | 영어 baseline |
| E1 | English | Predicted emotion prefix | emotion conditioning |
| E2 | English | Random emotion prefix | prefix control |
| E3 | English | Emotion SFT -> Sonnet SFT | representation transfer |
| K0 | Korean | KPoEM SFT only | 한국어 baseline |
| K1 | Korean | Predicted emotion prefix | emotion conditioning |
| K2 | Korean | Random emotion prefix | prefix control |
| K3 | Korean | KOTE SFT -> KPoEM SFT | representation transfer |

## 9. 평가 관련 주의

Main metric은 CHRF입니다. 다만 시 생성은 open-ended task이므로 CHRF만으로 시적 품질을 완전히 평가할 수 없습니다.

Report limitation에 넣을 내용:

```text
CHRF measures reference similarity, not overall poetic quality. A generated poem
can be fluent and emotionally consistent while receiving a low CHRF score if it
differs from the reference continuation.
```

가능하면 future work로 다음을 언급합니다.

```text
human preference evaluation
LLM-as-a-judge
emotion consistency
diversity metrics such as Distinct-n or self-BLEU
```

## 10. 현재 Git에서 추적하지 않는 데이터

다음 경로는 `.gitignore`되어 있습니다.

```text
data/raw/
data/processed/
```

따라서 clone 후 반드시 전처리 명령을 다시 실행해야 합니다.

