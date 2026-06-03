# 감정 정보를 활용한 GPT-2 기반 시 생성 실험 제안서

## 1. Introduction

이번 프로젝트의 목표는 GPT-2 계열 언어모델에서 **감정 분류 능력이 시 생성 성능에 긍정적인 영향을 줄 수 있는지** 실험하는 것이다. 기본 과제는 CS224n GPT-2 프로젝트처럼 GPT-2 구조를 구현하고, sentiment analysis, paraphrase detection, sonnet generation 세 가지 downstream task를 수행하는 것이다.

우리는 이 중 감정/감성 분류 task와 시 생성 task를 연결해 다음 질문을 다룬다.

> 감정 분류를 통해 학습된 감정 정보가 시의 나머지 부분을 생성하는 데 도움이 되는가?

기본 sonnet generation task는 시의 첫 3줄을 입력으로 받고, 나머지 줄을 생성하는 방식이다. 생성 결과는 실제 시와 비교하여 CHRF 같은 character-level similarity metric으로 평가할 수 있다. 우리는 여기에 감정 정보를 추가했을 때, 모델이 원문의 분위기나 정서를 더 잘 이어받을 수 있는지 확인하려고 한다.

## 2. Motivation

시 생성은 단순한 문장 생성보다 어렵다. 문법적으로 자연스러운 문장을 만드는 것뿐 아니라, 앞부분의 분위기, 정서, 이미지, 주제 흐름을 유지해야 하기 때문이다. 첫 세 줄이 슬픈 분위기인데 뒤에서 갑자기 밝고 가벼운 문장이 나오면 continuation으로는 어색하다.

사람이 시를 쓸 때도 감정 이해는 중요하다. 물론 GPT-2가 인간처럼 감정을 이해한다고 볼 수는 없지만, 감정 분류 task를 통해 감정과 관련된 표현 패턴을 학습할 수는 있다. 이 표현 패턴이 시 생성에도 도움이 되는지 확인하는 것이 이 실험의 핵심이다.

기본 과제에서는 sentiment analysis와 sonnet generation이 서로 독립된 task로 다뤄진다. 우리는 이 둘을 연결하여 다음 두 가지 가능성을 실험한다.

- 감정 label을 prompt에 명시적으로 넣으면 시 생성이 좋아지는가?
- 감정 분류 task로 먼저 fine-tuning한 뒤 시 생성 task로 fine-tuning하면 성능이 좋아지는가?

## 3. Research Questions

### Main Question

감정 supervision은 GPT-2 계열 모델의 poem continuation 성능을 향상시키는가?

### Subquestions

1. 예측된 감정 label을 prompt prefix로 넣으면 시 생성 성능이 향상되는가?
2. 감정 분류 fine-tuning 이후 시 생성 fine-tuning을 하면 성능이 향상되는가?
3. 이 효과가 영어 GPT-2와 한국어 KoGPT2에서 비슷하게 나타나는가?
4. 영어/한국어 감정 데이터와 시 데이터는 label 분포, 문장 길이, tokenization 측면에서 어떤 차이를 보이는가?

## 4. Datasets

### 4.1 영어 감정 분류 데이터셋

영어 감정 분류에는 `dair-ai/emotion`을 사용한다.

- Source: https://huggingface.co/datasets/dair-ai/emotion
- Language: English
- Domain: 짧은 Twitter 스타일 문장
- Labels: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`
- Format: single-label classification

이 데이터셋은 6개 감정 class가 명확하고, 현재 GPT-2 classifier 구조와 잘 맞는다. 한 문장에 하나의 label만 있으므로 cross-entropy loss로 바로 학습할 수 있다.

### 4.2 영어 시 생성 데이터셋

영어 시 생성에는 기본 CS224n GPT-2 프로젝트의 sonnet generation dataset을 사용한다.

- Input: sonnet 첫 3줄
- Target: 나머지 sonnet
- Task: poem continuation
- Evaluation: reference continuation과 생성 continuation의 CHRF

이 데이터셋은 영어 generation baseline으로 사용한다.

### 4.3 한국어 감정 분류 데이터셋

한국어 감정 분류에는 KOTE를 사용한다.

- Dataset: KOTE, Korean Online That-gul Emotions
- Source: https://github.com/searle-j/KOTE
- Paper: https://arxiv.org/abs/2205.05300
- Language: Korean
- Domain: 온라인 댓글
- Original labels: 43개 감정 label + `NO EMOTION`
- Original format: multi-label classification

KOTE는 원래 multi-label이고 class 수도 많다. 하지만 영어 감정 데이터셋인 `dair-ai/emotion`이 6-class single-label이므로, 비교 가능성을 위해 KOTE도 6-class single-label 데이터셋으로 변환한다.

### 4.4 한국어 시 생성 데이터셋

한국어 시 생성에는 KPoEM을 사용한다.

- Dataset: KPoEM, Korean Poetry Emotion Mapping Dataset
- Source: https://huggingface.co/datasets/AKS-DHLAB/KPoEM
- Project page: https://www.kadh.org/dataset-model-kpoem/
- Language: Korean
- Domain: 한국 근대시
- Size: 약 483편
- Available levels: line-level, poem-level
- Emotion labels: 44개 세부 감정 category

KPoEM은 원래 sonnet continuation용 데이터셋은 아니지만, poem-level text를 사용하면 기본 sonnet generation task와 비슷한 형태로 변환할 수 있다.

```text
input  = 시의 첫 3행
target = 4행 이후 나머지 시
```

KPoEM의 감정 label은 main 실험보다는 appendix 또는 추가 분석에 활용한다. main generation task는 시 본문을 first-3-lines continuation 형태로 변환해서 사용한다.

## 5. Preprocessing Plan

### 5.1 영어 감정 데이터셋

`dair-ai/emotion`은 전처리가 거의 필요 없다.

1. train/validation/test split을 불러온다.
2. label을 공통 6-class label set으로 정규화한다.
3. 다음 형식의 CSV 또는 JSONL로 저장한다.

```text
id,text,label
0,"I feel lonely tonight.",sadness
1,"I am so happy to see you.",joy
```

### 5.2 한국어 감정 데이터셋

KOTE의 43개 세부 감정을 영어 데이터셋과 같은 6개 class로 mapping한다.

공통 target label:

```text
sadness, joy, love, anger, fear, surprise
```

예상 mapping table:

| Target label | KOTE 세부 감정 |
|---|---|
| sadness | 슬픔, 서러움, 절망, 힘듦/지침, 안타까움/실망, 패배/자기혐오 |
| joy | 기쁨, 행복, 즐거움/신남, 감동/감탄, 뿌듯함, 고마움, 편안/쾌적 |
| love | 아껴주는, 환영/호의, 흐뭇함, 존경 |
| anger | 화남/분노, 짜증, 불평/불만, 증오/혐오, 지긋지긋 |
| fear | 공포/무서움, 불안/걱정, 의심/불신, 부담/안 내킴 |
| surprise | 놀람, 경악, 당황/난처, 신기함/관심, 깨달음 |

Filtering rule:

1. 각 KOTE sample의 원래 label들을 6개 target class로 mapping한다.
2. 정확히 하나의 target class에만 대응되는 sample은 keep한다.
3. 대응되는 target class가 없는 sample은 drop한다.
4. 여러 target class에 동시에 걸리는 sample은 main 실험에서는 drop한다.

이렇게 하면 영어와 한국어 모두 single-label 6-class emotion classification으로 맞출 수 있다.

### 5.3 한국어 시 생성 데이터셋

KPoEM에서는 poem-level 데이터를 사용한다.

전처리 단계:

1. poem-level TSV 파일을 불러온다.
2. 줄바꿈과 공백을 정규화한다.
3. 비어 있는 행을 제거한다.
4. 4행 미만의 시는 제거한다.
5. 각 시를 다음처럼 분리한다.

```text
prefix = 첫 3행
target = 4행 이후 나머지
full_text = 전체 시
```

6. 가능한 경우 metadata를 유지한다.

```text
id,title,poet,prefix,target,full_text,emotion_labels
```

7. line 단위가 아니라 poem 단위로 split한다.

추천 split:

```text
train/dev/test = 80/10/10
```

KPoEM이 약 483편이므로 대략 다음 정도가 된다.

```text
train: 약 386편
dev:   약 48편
test:  약 49편
```

데이터가 적긴 하지만, dev set을 완전히 없애기보다는 최소한의 dev set을 두는 것이 early stopping과 overfitting 확인에 유리하다. 모든 실험은 같은 split seed를 사용한다.

## 6. Models

### 6.1 영어 모델

영어 실험에는 현재 CS224n 프로젝트에서 구현한 GPT-2를 사용한다.

- Base model: GPT-2 small scale
- Approximate size: 124M parameters
- Architecture: decoder-only causal Transformer
- Tasks:
  - emotion classification
  - sonnet generation

### 6.2 한국어 모델

한국어 실험에는 `skt/kogpt2-base-v2`를 사용한다.

- Source: https://huggingface.co/skt/kogpt2-base-v2
- Model type: 한국어 GPT-2 계열 causal language model
- Approximate size: 125M parameters
- Architecture: decoder-only causal Transformer

한국어 GPT-2를 직접 pretraining하는 것은 프로젝트 범위를 지나치게 벗어나므로, 이미 공개된 GPT-2급 한국어 pretrained model을 사용하는 것이 현실적이다. `skt/kogpt2-base-v2`는 영어 GPT-2 small과 parameter 규모가 비슷해서 비교 대상으로 적절하다.

## 7. Method

우리는 감정 정보를 시 생성에 넣는 두 가지 방식을 비교한다.

### 7.1 Baseline: Poem SFT Only

baseline은 시 생성 데이터셋만 사용해서 fine-tuning한 모델이다.

영어:

```text
GPT-2 -> sonnet generation SFT
```

한국어:

```text
KoGPT2 -> KPoEM poem generation SFT
```

입력:

```text
시의 첫 3줄
```

출력:

```text
나머지 시
```

이 baseline은 감정 정보를 전혀 사용하지 않는 일반 poem continuation 성능을 측정한다.

### 7.2 Pipeline A: Emotion Prefix Prompting

첫 번째 방법은 감정 classifier가 첫 3줄의 감정을 예측하고, 그 감정 label을 generation prompt 앞에 붙이는 방식이다.

영어:

```text
first 3 lines
-> English emotion classifier
-> predicted emotion
-> [emotion: sadness] + first 3 lines
-> English poem generator
```

한국어:

```text
첫 3행
-> Korean emotion classifier
-> predicted emotion
-> [감정: 슬픔] + 첫 3행
-> Korean poem generator
```

이 방식은 명시적인 감정 conditioning이 시 생성에 도움이 되는지 확인한다.

추가 control로 random emotion prefix를 사용한다.

```text
[emotion: random_label] + first 3 lines
```

이 control은 성능 향상이 실제 감정 정보 때문인지, 아니면 단순히 prompt 앞에 token이 추가되었기 때문인지 구분하는 데 필요하다.

### 7.3 Pipeline B: Sequential Fine-Tuning

두 번째 방법은 모델을 감정 분류 task로 먼저 fine-tuning한 뒤, 같은 backbone을 시 생성 task로 다시 fine-tuning하는 방식이다.

영어:

```text
GPT-2
-> emotion classification SFT on dair-ai/emotion
-> sonnet generation SFT
```

한국어:

```text
KoGPT2
-> emotion classification SFT on collapsed KOTE
-> KPoEM poem generation SFT
```

이 방식은 감정 분류에서 학습한 representation이 generation task로 transfer되는지 확인한다.

classification과 generation은 head가 다르기 때문에, sequential SFT에서는 task-specific head가 아니라 shared GPT backbone을 이어받는다. 감정 분류 후에는 classification head를 제거하고, generation용 LM head로 시 생성 fine-tuning을 진행한다.

## 8. Experimental Matrix

| ID | Language | Model | Training / Input Setup | Purpose |
|---|---|---|---|---|
| E0 | English | GPT-2 | Sonnet SFT only | 영어 baseline |
| E1 | English | GPT-2 | Predicted emotion prefix + sonnet SFT | 명시적 감정 conditioning |
| E2 | English | GPT-2 | Random emotion prefix + sonnet SFT | prefix control |
| E3 | English | GPT-2 | Emotion SFT -> sonnet SFT | representation transfer |
| K0 | Korean | KoGPT2 | KPoEM SFT only | 한국어 baseline |
| K1 | Korean | KoGPT2 | Predicted emotion prefix + KPoEM SFT | 명시적 감정 conditioning |
| K2 | Korean | KoGPT2 | Random emotion prefix + KPoEM SFT | prefix control |
| K3 | Korean | KoGPT2 | KOTE emotion SFT -> KPoEM SFT | representation transfer |

## 9. Evaluation

### 9.1 Main Metric

main metric은 CHRF를 사용한다.

CHRF는 character n-gram overlap 기반 metric이므로, word-level tokenization이 애매한 한국어에도 비교적 적용하기 쉽다. 영어 sonnet generation과 한국어 poem generation 모두 reference continuation과 generated continuation을 비교하는 방식으로 평가한다.

### 9.2 Additional Analyses

추가로 다음 항목을 분석한다.

- emotion classifier accuracy
- generated poem length
- 영어 GPT-2 tokenizer와 한국어 KoGPT2 tokenizer 기준 token length
- emotion dataset label distribution
- baseline / emotion-prefix / random-prefix / sequential SFT 간 CHRF 차이
- 선택적으로 generated poem의 emotion consistency

## 10. Training Budget Control

영어와 한국어는 tokenizer가 다르기 때문에 같은 epoch 수만으로 비교하면 불공정할 수 있다. 같은 문장 수라도 실제 token 수가 다를 수 있기 때문이다.

따라서 다음 조건을 최대한 맞춘다.

- model scale: GPT-2 small과 KoGPT2 base
- max sequence length
- effective batch size
- optimizer step 수
- learning rate schedule
- 각 dataset의 split seed

가능하면 epoch 수보다는 **동일한 optimizer step 수** 또는 **동일한 token budget**을 기준으로 비교한다.

## 11. Expected Outcomes

가능한 결과는 크게 세 가지다.

1. Emotion prefix가 baseline보다 높은 CHRF를 보인다.
   - 명시적인 감정 conditioning이 poem continuation에 도움이 된다는 근거가 된다.

2. Sequential emotion SFT가 baseline보다 높은 CHRF를 보인다.
   - 감정 분류 supervision이 generation에 유용한 representation을 제공할 가능성을 보여준다.

3. Emotion-based method가 baseline보다 좋아지지 않는다.
   - 이것도 의미 있는 결과다. CHRF가 정서적 품질을 잘 반영하지 못했을 수 있고, emotion classifier의 domain mismatch가 컸을 수도 있으며, 작은 시 데이터셋에서는 모델이 감정보다 문체 암기에 더 의존했을 수도 있다.

## 12. Fixed Choices

| Component | Choice |
|---|---|
| 영어 emotion dataset | `dair-ai/emotion` |
| 한국어 emotion dataset | KOTE를 6-class single-label로 변환 |
| 한국어 GPT-2 model | `skt/kogpt2-base-v2` |
| 한국어 poem dataset | KPoEM poem-level data |
| 한국어 generation task | 첫 3행 -> 나머지 시 |
| Main methods | emotion prefix prompting, sequential emotion SFT |
| Main metric | CHRF |

## 13. Remaining Implementation Tasks

1. `dair-ai/emotion` 전처리 script 작성
2. KOTE 43-to-6 label mapping 및 filtering script 작성
3. KPoEM poem-level preprocessing script 작성
4. CHRF evaluation script 작성
5. 현재 GPT-2 classifier를 6-class emotion classification에 맞게 수정
6. KoGPT2 loading 및 fine-tuning code 추가
7. emotion-prefix generation 구현
8. sequential SFT checkpoint transfer 구현
9. 작은 subset으로 CPU smoke test 수행
10. GPU 확보 후 full experiment 수행

