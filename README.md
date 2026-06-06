# 감정 정보 기반 GPT-2 시 생성 프로젝트

CS224n 스타일로 직접 구현한 GPT-2(영어)와 KoGPT2(한국어)를 사용해, **"감정 분류로 학습된 감정 정보가 시 continuation 생성에 도움이 되는가?"** 를 두 라운드에 걸쳐 검증한 프로젝트입니다.

> **최종 결론 (요약):** 감정 라벨 prefix, 감정 분류 → 시 생성 순차 fine-tuning, 행 단위 감정 흐름(trajectory) 가이드, prefix 비율 확대, 정답 sentiment 곡선 제공 — **어떤 방식으로도 시 continuation 품질(CHRF·BERTScore·감정 궤적 지표)이 개선되지 않았습니다.** 정답 감정과 무작위 감정의 차이가 없었다는 점이 핵심 증거입니다. 유일한 예외적 단서는 김소월 시에 한정했을 때 oracle 감정 흐름이 모든 통제군을 앞선 것(n=11, 시사적 수준)입니다. 상세는 `docs/`의 보고서 PDF 참조.

---

## 1. 저장소 구조

```text
├── README.md                  ← 이 문서
├── HANDOFF.md                 데이터 담당 → 학습 담당 인수인계 규격 (Round 2 실행 우선순위)
├── proposal.md                Round 1 실험 제안서 원문
│
├── models/, modules/          직접 구현한 GPT-2 본체 (GPT2Model, attention, layer)
├── config.py, utils.py        GPT-2 설정/유틸 (CS224n 스타터)
├── optimizer.py               직접 구현한 AdamW
├── datasets.py                기본 task 데이터 로더 (SonnetsDataset 등)
├── classifier.py              [기본 task] SST/CFIMDB 감성 분류
├── paraphrase_detection.py    [기본 task] Quora 패러프레이즈 탐지
├── sonnet_generation.py       [기본 task] 소네트 생성 (SonnetGPT — 시 생성 실험이 재사용)
├── evaluation.py, sanity_check.py, prepare_submit.py, optimizer_test.py
│
├── experiments/               모델-비의존 글루 레이어 (라이브러리, 직접 실행 X)
│   ├── data_bridge.py         processed JSONL 로더 (EmotionJsonlDataset, PoemContinuationJsonlDataset)
│   ├── prompts.py             감정 prefix 프롬프트 빌더 ([emotion: X] / [감정: X])
│   ├── metrics.py             CHRF, Distinct-n, 감정궤적 MSE/Corr, Volta distance
│   ├── emotion_predict.py     학습된 감정 분류기 로드 + 텍스트 감정 예측
│   ├── line_measure.py        행 단위 감정/sentiment 측정기 (EN: SST-2 DistilBERT, KO: 자체 분류기)
│   └── kpoem_emotion.py       KPoEM 세부감정 44종 → 공통 6-class 변환 (strict/poetic 매핑)
│
├── scripts/                   데이터 준비 스크립트 (모두 1회성, CPU)
│   ├── prepare_external_data.py              외부 3개 데이터셋 다운로드+전처리 (최초 1회 필수)
│   ├── smoke_test_data_bridge.py             데이터/글루 레이어 동작 검증
│   ├── prepare_kpoem_oracle_emotion.py       KPoEM poem 단위 oracle 감정 라벨 데이터
│   ├── prepare_kpoem_line_trajectory.py      KPoEM 행 단위 감정 흐름 데이터 (--prefix_ratio 지원)
│   ├── prepare_shakespeare_sentiment_trajectory.py  Shakespeare 14행 sentiment 곡선 데이터
│   └── run_grid.sh                           Round 2 전체 그리드 실행 파이프라인
│
├── training/                  실험 실행 스크립트 (레포 루트에서 실행)
│   ├── train_emotion.py       감정 분류기 fine-tuning (영/한)
│   ├── train_poem.py          시 생성 fine-tuning (베이스라인/prefix/순차전이/trajectory 전부 커버)
│   ├── evaluate_generation.py 학습된 모델로 continuation 생성 + CHRF 측정 → results/gen_*.jsonl
│   └── score_generations.py   생성물 풀 지표 재채점 (BERTScore·감정궤적 등) → results/scored_*
│
├── analysis/                  결과 집계·보고서
│   ├── analyze_results.py     Round 1 집계 → results/analysis.json
│   ├── analyze_grid.py        Round 2 그리드 집계 + paired 비교 → results/grid_analysis.json
│   ├── make_report.py         Round 1 보고서 PDF 생성 → docs/
│   ├── make_report_grid.py    Round 2 보고서 PDF 생성 → docs/
│   └── build_report.py        기본 3 task 보고서 PDF 생성 → docs/
│
├── checkpoints/               학습된 모델 (.pt는 git 미추적)
│   ├── base_tasks/            기본 3 task 체크포인트 (구 pt/ 폴더)
│   ├── emotion/               감정 분류기 2종 + dev 평가 결과(.eval.json)
│   ├── round1_emotion_prefix/ Round 1 시 생성 모델 8종
│   └── round2_trajectory/     Round 2 시 생성 모델 16종
│
├── data/                      sonnets.txt 등 기본 데이터 + data/processed/ (재생성 필요, git 미추적)
├── results/                   생성물(gen_*), 채점(scored_*), 집계(analysis/grid_analysis.json)
├── logs/                      학습/그리드 로그
├── docs/                      실험 설계 PDF 2편 + 결과 보고서 PDF 3편
└── predictions/               기본 task 제출물
```

---

## 2. 환경 설정

| 항목 | 값 |
|---|---|
| conda 환경 | `NLP_Project` (주의: `env.yml`의 `cs224n_dfp`는 사용하지 않음) |
| PyTorch | 2.11.0+cu128 (RTX 5070 Ti 17GB에서 검증) |
| 주요 패키지 | transformers 4.46.3, sacrebleu 2.5.1, bert-score 0.3.13, pandas, pyarrow, einops, scikit-learn |

**Windows 인코딩 주의:** `conda run`은 한글 출력 시 cp949 크래시가 납니다. 반드시 환경의 python을 직접 호출하세요.

```bash
PYTHONIOENCODING=utf-8 "C:/Users/<user>/anaconda3/envs/NLP_Project/python.exe" <script> ...
```

이하 명령 예시는 `python`으로 줄여 씁니다. **모든 스크립트는 레포 루트에서 실행합니다.**

---

## 3. 데이터 준비 (clone 후 최초 1회)

외부 데이터는 git에 올리지 않으므로 재생성이 필요합니다. 순서대로:

```bash
python scripts/prepare_external_data.py            # ① 감정(en/ko) + 시(ko) 기본 JSONL
python scripts/smoke_test_data_bridge.py           # ② 검증 (en_train 16000 / ko_train 11880 / poem_ko_train 368, CHRF identity 100.0)
python scripts/prepare_kpoem_oracle_emotion.py     # ③ KPoEM poem 단위 oracle 감정
python scripts/prepare_kpoem_line_trajectory.py    # ④ 행 단위 감정 흐름 (첫 3행 기준)
python scripts/prepare_kpoem_line_trajectory.py --prefix_ratio 0.3   # ⑤ ratio 30/50/70 데이터
python scripts/prepare_kpoem_line_trajectory.py --prefix_ratio 0.5
python scripts/prepare_kpoem_line_trajectory.py --prefix_ratio 0.7
python scripts/prepare_shakespeare_sentiment_trajectory.py --continuous  # ⑥ Shakespeare sentiment 곡선
```

- 데이터 출처: 감정(영) `dair-ai/emotion`, 감정(한) KOTE→6-class strict 변환, 시(한) KPoEM 461편, 시(영) 셰익스피어 소네트 154편(전부), sentiment 곡선 Zenodo "Shakespeare's Sonnets Volta & Sentiment".
- split: poem 단위 80/10/10, **seed 11711 고정**.
- ⑥의 `--continuous` 필수: 정수 반올림(discrete) 시 train 평균 곡선이 거의 0으로 뭉개집니다.
- 김소월 부분집합(`*_sowol.jsonl`)은 위 파일에서 poet 필드로 필터한 것입니다(131/18/11편).

---

## 4. 실험 흐름 한눈에 보기

```text
[기본 3 task]  감성분류·패러프레이즈·소네트생성 (CS224n)          → docs/NLP_GPT2_실험_보고서.pdf
      │
[공통 준비]   감정 분류기 fine-tuning (영 GPT-2 / 한 KoGPT2)     → checkpoints/emotion/
      │
[Round 1]    "감정 정보를 주면 시 continuation이 좋아지는가?"
      │        방식A: 예측 감정 라벨을 prefix 한 줄로 부착
      │        방식B: 감정분류 → 시생성 순차 fine-tuning
      │        결과: 전부 효과 없음 (예측 감정 = 무작위 감정)     → docs/감정기반_시생성_실험_보고서.pdf
      │
[Round 2]    "라벨이 거칠어서 실패했나? 행 단위 감정 '흐름'이면 다른가?" (진단)
               행별 감정 나열 prefix(oracle/평균/무작위) × prefix 비율 30/50/70%
               + 영어 14행 sentiment 곡선 + 김소월 특화
               결과: 역시 효과 없음, 김소월 한정 단서만           → docs/감정흐름_시생성_후속실험_보고서.pdf
```

---

## 5. 공통 실험 규약 (모든 실험에 적용)

비교의 전제는 **"비교 대상끼리는 감정 정보 외 모든 조건 동일"** 입니다.

| 항목 | 값 (default) | 그렇게 정한 이유 |
|---|---|---|
| random seed | **11711** | CS224n 스타터 기본값. 데이터 split·학습 초기화·디코딩 샘플링까지 전부 고정해 재현성 확보 |
| optimizer | 직접 구현 AdamW | 기본 task와 동일 구현 사용 (일관성) |
| learning rate | **1e-5** | full-model fine-tuning 표준값. 더 크면 사전학습 지식이 파괴됨(스타터 가이드의 finetune 권장값) |
| fine-tune 범위 | **full-model** (전 파라미터) | 순차 전이 실험이 의미 있으려면 backbone 자체가 갱신되어야 함 |
| 체크포인트 선택 | **dev 성능 best 시점 저장** | 한국어 시 데이터(368편)는 1~2 epoch 만에 과적합으로 dev loss가 반전 상승 → best-dev 저장이 자동 가드 |
| 영어 backbone | 직접 구현 GPT2Model + gpt2 가중치 (124M) | 기본 task와 동일 모델 |
| 한국어 backbone | HuggingFace `skt/kogpt2-base-v2` 원본 (125M) | GPT-2 small과 구조 동일(768/12/12)이라 비교 가능. 한국어 원본 세팅 보존을 위해 영/한 코드 경로 분리 |

**KoGPT2 함정:** tokenizer는 반드시 special token을 명시해 로드해야 합니다(`pad=<pad>(3), eos=</s>(1)`). 자동 로드 시 eos id 51200이 임베딩 범위(0~51199)를 벗어나 crash합니다. (`training/train_emotion.py::build_tokenizer` 참조)

---

## 6. 단계별 실험 상세

### 6.1 감정 분류기 학습 — `training/train_emotion.py`

**무엇을 하나:** 사전학습 GPT-2/KoGPT2 위에 6-class(슬픔·기쁨·사랑·분노·공포·놀람) 분류 head를 붙여 fine-tuning. 마지막 비-패딩 토큰의 hidden state → Linear(768→6). 이 분류기는 이후 (a) 시 첫 부분의 감정 예측(prefix 실험), (b) backbone 전이(순차 fine-tuning), (c) 생성물의 행별 감정 측정(채점기) 세 곳에서 재사용됩니다.

```bash
python training/train_emotion.py --lang en --use_gpu --epochs 3 --batch_size 32
python training/train_emotion.py --lang ko --use_gpu --epochs 3 --batch_size 32
```

| 하이퍼파라미터 | 사용값 (default) | 이유 |
|---|---|---|
| epochs | **3** (3) | dev macro-F1이 3 epoch 내 수렴 (영 0.899 / 한 0.704). 그 이상 이득 없음 |
| batch_size | **32** (16) | max 64토큰의 짧은 문장이라 17GB VRAM에 충분. 학습 속도 위해 32로 상향 |
| lr | **1e-5** (1e-5) | 공통 규약 |
| max_length | **64** (64) | 트윗/댓글 문장 길이 분포를 여유 있게 커버 |
| hidden_dropout_prob | **0.3** (0.3) | 원본 `classifier.py` 기본값 유지 |
| weight_mode / weight_cap | **sqrt / 5.0** (sqrt / 5.0) | 한국어는 anger가 48.6%로 쏠려 있어 보정 없으면 전부 anger로 찍는 다수클래스 붕괴 발생 → prefix 실험 자체가 무의미해짐. 단순 역빈도는 329개짜리 소수클래스(fear)에 과대 가중되어 불안정 → **sqrt 역빈도로 완화 + 상한 5.0** |
| 모델 선택 지표 | **dev macro-F1** | 불균형 데이터에서 accuracy는 다수클래스에 편향됨 |

**결과:** 영어 acc 0.917 / macro-F1 0.899, 한국어 acc 0.820 / macro-F1 0.704. 다수클래스 붕괴 없음(6class 모두 대각 우세). 상세 per-class F1·confusion matrix는 `checkpoints/emotion/*.eval.json`.

### 6.2 Round 1 — 감정 prefix와 순차 fine-tuning

**연구 질문:** 감정 정보를 (A) 입력 텍스트로 주거나 (B) 초기 가중치로 주면 시 continuation(첫 3행 → 나머지)이 좋아지는가?

**실험 구성** (파일명의 이니셜 ↔ 실제 내용):

| 체크포인트 | 실험 내용 |
|---|---|
| `poem_en_E0.pt` / `poem_ko_K0.pt` | **베이스라인.** 시 본문만으로 LM fine-tuning. 감정 정보 전혀 미사용 |
| `poem_en_E1.pt` / `poem_ko_K1.pt` | **예측 감정 prefix.** 감정 분류기가 각 시 첫 3행에서 예측한 감정 라벨을 `[emotion: X]`/`[감정: X]` 한 줄로 시 앞에 붙여 **학습하고, 생성 시에도 같은 방식으로 부착** (학습 때 prefix를 봐야 모델이 활용법을 배우므로 베이스라인 모델 재사용이 아닌 별도 모델) |
| `poem_en_E2.pt` / `poem_ko_K2.pt` | **무작위 감정 prefix (통제군).** 예측 대신 무작위 라벨. 예측-prefix가 좋아져도 그것이 '감정 내용' 덕인지 '아무 prefix나 붙은 효과'인지 가리는 장치 |
| `poem_en_E3.pt` / `poem_ko_K3.pt` | **순차 fine-tuning (sequential SFT).** 감정 분류로 fine-tuning된 분류기에서 GPT backbone(`gpt.*`)만 떼어와 초기값으로 쓰고 시 생성으로 이어서 fine-tuning. prefix 없음 |

```bash
# 베이스라인 / 순차 전이 / 예측 prefix / 무작위 prefix (한국어 예시; 영어는 --lang en)
python training/train_poem.py --lang ko --use_gpu
python training/train_poem.py --lang ko --init_from_emotion checkpoints/emotion/emotion_ko_classifier.pt --setup_id K3 --use_gpu
python training/train_poem.py --lang ko --emotion_prefix pred --emotion_ckpt checkpoints/emotion/emotion_ko_classifier.pt --setup_id K1 --use_gpu
python training/train_poem.py --lang ko --emotion_prefix random --setup_id K2 --use_gpu
# 평가 (예: 베이스라인)
python training/evaluate_generation.py --lang ko --ckpt checkpoints/round1_emotion_prefix/poem_ko_K0.pt --setup_id K0 --use_gpu --n_samples 3
```

**학습 하이퍼파라미터 — `training/train_poem.py`:**

| 하이퍼파라미터 | 사용값 (default) | 이유 |
|---|---|---|
| epochs | **10** (10) | 영어는 10 내 dev loss 단조 개선, 한국어는 best-dev 가드 하에 충분한 상한 |
| batch_size | **8** (8) | 원본 `sonnet_generation.py` 기본값과 동일(기본 task와의 일관성) + 시 전체 시퀀스가 길어 메모리 고려 |
| lr | **1e-5** (1e-5) | 공통 규약 |
| max_length (한국어) | **256** (256) | KPoEM 시 평균 ~168토큰 — 여유 커버 |
| 학습 목표 | full-text LM (모든 비-패딩 토큰에 cross-entropy, 패딩은 -100 무시) | HANDOFF 권장 — 베이스라인은 가장 단순한 형태. prefix-마스킹 변형은 future work |

**디코딩/평가 하이퍼파라미터 — `training/evaluate_generation.py` (모든 실험·라운드 공통):**

| 하이퍼파라미터 | 사용값 (default) | 이유 |
|---|---|---|
| temperature / top_p | **0.8 / 0.9** (0.8 / 0.9) | nucleus 샘플링. 스타터 기본 1.2는 과도하게 발산해 비교 노이즈가 큼 → 다양성·일관성 절충. **모든 setup 동일이 비교의 전제** |
| n_samples | **3** (3) | 평가셋이 작아(영 12편/한 47편) 단일 샘플은 분산이 큼 → 시당 3회 생성해 점수 평균. 시드는 (기준시드 + 샘플번호×1000 + 시번호)로 고정 |
| max_new_tokens | **200** (200) | 영어 11행/한국어 평균 continuation 길이 커버. 전 setup 동일 상한 |
| no_repeat_ngram_size / repetition_penalty (한국어만) | **3 / 1.3** (3 / 1.3) | KoGPT2가 plain 샘플링에서 같은 구절 무한 반복("백골만 따라갔다"×N)에 빠져 CHRF가 바닥(~3)이 됨 → 반복 억제. 모든 한국어 setup 동일 적용으로 공정성 유지. 영어는 자체 디코더로 문제 없어 미적용 |
| 평가셋 | 영어 dev 12편(gold 보유) / 한국어 test 47편 | 영어 실제 test 12편은 gold 비공개(CS224n 제출용). 셰익스피어 소네트는 154편이 전부라 원래 작음 — 같은 시에 모든 setup을 적용하는 paired 비교로 보완 |

**결과 (continuation CHRF):** 영어 — 베이스라인 27.81 / 예측prefix 26.57 / 무작위prefix 26.41 / 순차전이 23.67. 한국어 — 7.16 / 7.17 / 7.27 / 7.26.
**결론:** ① 예측 감정 ≈ 무작위 감정(영 Δ+0.17, 6:6 무승부) → 감정 내용 무신호. ② 순차 전이는 영어에서 12편 중 11편 악화(평균 -4.14) — 트윗/댓글 도메인으로 fine-tuning된 backbone이 시적 능력을 잠식. ③ 한국어는 전부 노이즈 수준.

### 6.3 Round 2 — 행 단위 감정 흐름(trajectory) 진단

**연구 질문:** Round 1 실패가 "시 전체에 라벨 1개"라는 거친 표현 때문이었는가? 행마다 감정을 나열한 **구조화된 감정 흐름**을 주면 달라지는가? (미래 행 감정은 절대 입력하지 않음 — leakage-safe)

**실험 구성** (파일명 ↔ 실제 내용):

| 체크포인트/세팅 | 실험 내용 |
|---|---|
| `K_base` (재평가) | Round 1 한국어 베이스라인 모델 그대로, 새 지표로 재평가 |
| `poem_ko_K_orc.pt` | **oracle 감정 흐름.** prefix 각 행의 KPoEM 정답 감정(주석자 다수결, poetic 매핑)을 `[감정흐름] 1: 슬픔 2: 기쁨 ...` 블록으로 시 앞에 부착해 학습·생성 |
| `poem_ko_K_avg.pt` | **평균 감정 흐름.** train 전체에서 위치별 다수결로 만든 고정 흐름을 모든 시에 동일 부착 (정답을 모를 때 쓸 수 있는 prior) |
| `poem_ko_K_rnd.pt` | **무작위 감정 흐름 (통제군).** 행별 무작위 라벨 |
| `poem_ko_K_r{30,50,70}_{orc,avg,rnd}.pt` | **prefix 비율 변형.** 고정 "첫 3행" 대신 시 길이의 30/50/70%를 prefix로 재분할 (긴 문맥일수록 흐름 정보가 많아지는지 검증). 비율별 베이스라인은 K0 모델을 해당 비율 입력으로 재평가 |
| `poem_en_E_orc.pt` | **영어 oracle sentiment 곡선 (Method A).** Zenodo 데이터셋의 정답 14행 sentiment 연속값을 `[sentiment trajectory] 1: +0.230 ...` 블록으로 부착 — "정답 곡선을 알면 좋아지는가" upper-bound |
| `poem_en_E_avg.pt` | **영어 평균 곡선 (Method B).** train 130편 평균 곡선을 모두에게 부착 |
| `poem_ko_S_base.pt` / `poem_ko_S_orc.pt` | **김소월 특화.** KPoEM 중 김소월 시 131편만으로 학습한 베이스라인/oracle 모델 (평가는 김소월 test 11편) |

전체 그리드 재현: `bash scripts/run_grid.sh` (16개 학습 + 21개 평가 일괄 실행)

**Round 1과 다른 하이퍼파라미터** (나머지는 §6.2와 동일):

| 하이퍼파라미터 | 사용값 | 이유 |
|---|---|---|
| max_length (한국어 ratio 학습) | **512** | `[감정흐름]` 블록이 prefix 행 수만큼 길어짐(70%에서 최대 151행) → 256이면 시 본문이 잘림 |
| max_length (영어 trajectory 학습) | **384** | 소네트 ~160토큰 + 14행 sentiment 블록(~150토큰) |
| Shakespeare 곡선 표현 | 연속값(`--continuous`), 소수 3자리 | 정수 반올림 시 평균 곡선이 0으로 뭉개져 Method B가 무의미해짐 |

**평가 지표 (full tier)** — `training/score_generations.py`로 채점:

| 지표 | 측정 대상 | 방향 |
|---|---|---|
| CHRF | 문자 n-gram 표면 유사도 (sacrebleu) | ↑ |
| BERTScore F1 | 의미 유사도 (영 roberta-large / 한 mBERT) | ↑ |
| Trajectory MSE / Corr | 행별 감정(valence ±1/0) 궤적의 거리/모양 일치 | ↓ / ↑ |
| Volta distance | 감정 전환점(최대 변화 행) 위치 차이 | ↓ |
| 행감정 일치율 (한국어) | 행별 6-class 라벨 일치 | ↑ |
| Distinct-1/2 | 반복 없는 다양성 | ↑ |

측정기: 영어 = 사전학습 SST-2 DistilBERT(외부 모델 → leakage 없음), 한국어 = §6.1의 자체 분류기를 행별 적용. **생성물과 정답에 같은 측정기를 적용해 측정기 편향을 상쇄**합니다.

**결과 요약:**
- 한국어: 모든 설정에서 CHRF·BERTScore flat. trajectory corr은 **무작위(0.088)가 oracle(0.060)보다 높음** — 행 단위로 구조화해도 감정 '내용'은 무신호. 비율 30/50/70% 어디서도 oracle이 무작위를 일관되게 못 이김.
- 영어: 정답 곡선이 오히려 **12편 전부에서 CHRF·BERTScore 악화**. 평균 곡선(Method B)의 Volta distance 개선(4.50→3.42)만 약한 구조 prior 가능성.
- 김소월: test 11편 한정 시 oracle이 모든 통제군 위 (traj_corr +0.199 > rnd +0.127 > avg +0.01 > base -0.173) — **유일한 긍정적 단서**, 단 n=11. 김소월 전용 학습 자체는 데이터 축소 손실이 더 커서 역효과.

---

## 7. 전체 재현 명령 모음

```bash
# 0) 데이터 (§3의 ①~⑥)
# 1) 감정 분류기
python training/train_emotion.py --lang en --use_gpu --epochs 3 --batch_size 32
python training/train_emotion.py --lang ko --use_gpu --epochs 3 --batch_size 32
# 2) Round 1 학습 (베이스라인→예측prefix→무작위prefix→순차전이, 영/한 각각. §6.2 명령 참조)
# 3) Round 1 평가 + 집계 + 보고서
python training/evaluate_generation.py --lang en --ckpt checkpoints/round1_emotion_prefix/poem_en_E0.pt --setup_id E0 --use_gpu   # (E0~E3, K0~K3 반복)
python analysis/analyze_results.py
python analysis/make_report.py
# 4) Round 2 전체 그리드 (학습+생성)
bash scripts/run_grid.sh
# 5) 풀 지표 채점 (모든 gen 파일에 대해 반복)
python training/score_generations.py --gen_file results/gen_K_orc.jsonl --lang ko --use_gpu
# 6) 집계 + 보고서
python analysis/analyze_grid.py
python analysis/make_report_grid.py
```

## 8. 결과물 위치

| 무엇 | 어디 |
|---|---|
| 실험 설계 문서 | `proposal.md`(Round 1), `docs/Emotion-Guided Korean Poetry Generation.pdf`·`docs/Shakespeare Sonnet Generation Experiment.pdf`(Round 2 설계) |
| 결과 보고서 | `docs/감정기반_시생성_실험_보고서.pdf`(Round 1), `docs/감정흐름_시생성_후속실험_보고서.pdf`(Round 2), `docs/NLP_GPT2_실험_보고서.pdf`(기본 3 task) |
| 시별 생성물/점수 | `results/gen_*.jsonl`(생성+CHRF), `results/scored_*.jsonl`(풀 지표) |
| 집계 | `results/analysis.json`(R1), `results/grid_analysis.json`(R2: 그룹 표+paired+김소월 교차) |
| 학습 로그 | `logs/` |

## 9. 한계

- CHRF·BERTScore는 reference 유사도이지 시적 품질이 아님. 정서적으로 잘 이어진 시도 reference와 다르면 점수가 낮음.
- 감정 분류기는 트윗/댓글 도메인 학습 → 시에 적용 시 도메인 불일치 (한국어 trajectory 측정에도 같은 노이즈 유입).
- 평가셋이 작음 (영 12 / 한 47 / 김소월 11편) → 절대값보다 paired 상대 비교 중심으로 해석.
- 예측 planner(첫 부분으로 미래 감정 곡선을 예측해 주입하는 Method C)는 scope 외로 남김.
