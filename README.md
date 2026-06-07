# GPT-2 시 생성 프로젝트: 기본 Downstream Task + 감정 곡선 조건화 최종 실험

직접 구현한 GPT-2(영어, 124M)와 KoGPT2(한국어)로 **① 기본 3개 downstream task**(감성 분류·패러프레이즈 탐지·소네트 생성)를 수행하고, **② 최종 실험**으로 "시의 정답 감정 곡선 전체를 제공하면 시 continuation 생성이 좋아지는가"를 검증한 프로젝트입니다.

> **최종 실험 결론:** 정답 감정 곡선 전체(미래 행 포함)를 학습·추론 양쪽에 제공해도 — 영어 3가지 sentiment 측정 체계, 한국어 44종/6종 라벨 해상도 모두에서 — 시 continuation 품질(chrF·BERTScore)은 무조건화 베이스라인을 넘지 못했다. 정답 곡선은 셔플 곡선과 구별되지 않았고(내용 무신호), 한국어에서는 정답 감정열을 받은 모델이 오히려 감정 흐름을 덜 따라가는 역설이 관측됐다. 유일하게 재현되는 효과는 train 평균 곡선에 의한 감정 전환점(Volta) 위치 개선으로, 시별 내용이 아닌 구조 prior 효과다. 상세: `docs/감정곡선_Round3_실험_보고서.pdf`

---

## 1. 저장소 구조 및 파일 설명

```text
├── README.md                  ← 이 문서
├── HANDOFF.md, proposal.md    실험 인수인계 규격 / 초기 제안서
│
│  [GPT-2 구현 및 기본 task — Task 1]
├── models/, modules/          직접 구현한 GPT-2 본체 (embedding, attention, layer, LM head)
├── config.py, utils.py, optimizer.py   설정 / 유틸 / 자체 구현 AdamW
├── datasets.py                기본 task 데이터 로더 (SonnetsDataset 등)
├── classifier.py              [Task 1] SST(5-class)·CFIMDB(2-class) 감성 분류
├── paraphrase_detection.py    [Task 1] Quora 패러프레이즈 탐지 (cloze 방식)
├── sonnet_generation.py       [Task 1] 소네트 생성 (SonnetGPT — 최종 실험이 재사용)
├── evaluation.py, sanity_check.py, prepare_submit.py
│
│  [실험 공용 레이어]
├── experiments/               모델-비의존 글루 (라이브러리, 직접 실행 X)
│   ├── data_bridge.py         processed JSONL 로더
│   ├── prompts.py             감정 prefix/블록 프롬프트 빌더
│   ├── metrics.py             chrF·Distinct-n·감정궤적 MSE/Corr·Volta distance
│   ├── emotion_predict.py     학습된 감정 분류기 로드·예측
│   ├── line_measure.py        행 단위 감정/sentiment 측정기 (EN: SST-2 DistilBERT, KO: 자체 분류기)
│   └── kpoem_emotion.py       KPoEM 세부감정 44종 → 6-class 매핑
│
│  [데이터 준비 — 모두 1회성 CPU 스크립트]
├── scripts/
│   ├── prepare_external_data.py             외부 데이터셋 다운로드+전처리 (최초 1회 필수)
│   ├── smoke_test_data_bridge.py            데이터/글루 검증
│   ├── prepare_kpoem_oracle_emotion.py      KPoEM poem 단위 감정 라벨
│   ├── prepare_kpoem_line_trajectory.py     KPoEM 행 단위 감정 흐름 (--prefix_ratio 지원)
│   ├── prepare_kpoem_fulltraj_fine.py       [최종 실험-한국어] 전행(全行) 감정 라벨열 (--label_mode fine/six)
│   ├── prepare_shakespeare_sentiment_trajectory.py  [최종 실험-영어] 14행 sentiment 곡선 (--score_col m/v/r)
│   ├── prepare_round3_en.py                 [최종 실험-영어] dev#138 제외 + 셔플 통제군 + plain base 생성
│   ├── run_round3.sh                        [최종 실험-영어] 전체 그리드 실행 파이프라인
│   └── run_grid.sh                          (중간 실험 파이프라인 — 기록 보존용)
│
│  [학습·평가 실행 — 레포 루트에서 실행]
├── training/
│   ├── train_emotion.py       감정 분류기 fine-tuning (영/한)
│   ├── train_poem.py          시 생성 fine-tuning (--train_file로 임의 조건화 데이터 학습)
│   ├── evaluate_generation.py 생성 + chrF 측정 → results/gen_*.jsonl
│   └── score_generations.py   풀 지표 재채점 (BERTScore·Volta 등) → results/scored_*
│
│  [분석·보고서]
├── analysis/
│   ├── analyze_round3.py      [최종 실험] 그리드 집계 + paired 비교 → results/round3_analysis.json
│   ├── make_report_round3.py  [최종 실험] 보고서 PDF 생성 → docs/
│   ├── build_report.py        [Task 1] 보고서 PDF 생성 → docs/
│   └── analyze_results.py, analyze_grid.py, make_report.py, make_report_grid.py  (중간 실험 분석 — 기록 보존용)
│
│  [산출물]
├── checkpoints/               학습된 모델 (.pt는 git 미추적)
│   ├── base_tasks/            [Task 1] 체크포인트 15개
│   ├── emotion/               감정 분류기 2종 (+dev 평가 .eval.json)
│   ├── round3_curve/          [최종 실험] 모델 12개 (영 10 + 한 2)
│   └── round1_emotion_prefix/, round2_trajectory/   (중간 실험 — 기록 보존용)
├── data/                      소네트 원문 + data/processed/ (재생성 필요, git 미추적)
├── results/                   gen_*(생성), scored_*(채점), round3_analysis.json(집계)
├── predictions/               [Task 1] 제출물 (감성/패러프레이즈 예측 CSV, 생성 소네트)
├── docs/                      보고서 PDF (Task 1, 중간 실험 2편, 최종 실험) + 설계 문서 2편
└── logs/                      학습 로그 (git 미추적)
```

---

## 2. 환경 설정

| 항목 | 값 |
|---|---|
| conda 환경 | `NLP_Project` (env.yml의 `cs224n_dfp`는 사용하지 않음) |
| PyTorch | 2.11.0+cu128 (RTX 5070 Ti 17GB 검증 — cu118은 sm_120 미지원) |
| 주요 패키지 | transformers 4.46.3, sacrebleu, bert-score, pandas, pyarrow, einops, scikit-learn |

**Windows 주의:** `conda run`은 한글 출력에서 cp949 크래시 → 환경 python 직접 호출:

```bash
PYTHONIOENCODING=utf-8 "C:/Users/<user>/anaconda3/envs/NLP_Project/python.exe" <script> ...
```

이하 `python`으로 표기. **모든 스크립트는 레포 루트에서 실행.**

---

## 3. 실행 방법

### 3.1 데이터 준비 (clone 후 최초 1회)

```bash
python scripts/prepare_external_data.py            # ① 감정(en/ko)+시(ko) 기본 JSONL
python scripts/smoke_test_data_bridge.py           # ② 검증 (16000/11880/368, CHRF identity 100)
# --- 최종 실험용 ---
python scripts/prepare_shakespeare_sentiment_trajectory.py --continuous                          # ③ 영어 곡선 m
python scripts/prepare_shakespeare_sentiment_trajectory.py --continuous --score_col sentiment_v  # ④ 영어 곡선 v
python scripts/prepare_shakespeare_sentiment_trajectory.py --continuous --score_col sentiment_r  # ⑤ 영어 곡선 r
python scripts/prepare_round3_en.py                # ⑥ dev#138 제외 + 셔플 통제군 + plain base
python scripts/prepare_kpoem_fulltraj_fine.py --label_mode fine   # ⑦ 한국어 전행 44종 라벨
python scripts/prepare_kpoem_fulltraj_fine.py --label_mode six    # ⑧ 한국어 전행 6종 라벨
```

### 3.2 Task 1 (기본 3 task) 실행

```bash
python sanity_check.py                       # GPT-2 구현 검증
python classifier.py --use_gpu --fine-tune-mode full-model   # SST+CFIMDB 감성 분류
python paraphrase_detection.py --use_gpu     # Quora 패러프레이즈
python sonnet_generation.py --use_gpu        # 소네트 생성
python analysis/build_report.py              # Task 1 보고서 PDF
```

### 3.3 최종 실험 실행

```bash
# 영어: base 1 + (oracle/average/shuffled × m/v/r) 9 = 10개 학습+평가 일괄
bash scripts/run_round3.sh

# 한국어: 전행 라벨 모델 학습 (fine 예시; six는 파일명의 fine→six)
python training/train_poem.py --lang ko \
  --train_file data/processed/poem_ko_train_fulltraj_fine_r30.jsonl \
  --dev_file   data/processed/poem_ko_dev_fulltraj_fine_r30.jsonl \
  --setup_id R3K_fine --use_gpu --epochs 10 --batch_size 8 --max_length 768 \
  --out checkpoints/round3_curve/poem_ko_R3K_fine.pt
python training/evaluate_generation.py --lang ko \
  --ckpt checkpoints/round3_curve/poem_ko_R3K_fine.pt --setup_id R3K_fine \
  --eval_file data/processed/poem_ko_test_fulltraj_fine_r30_sowol.jsonl --use_gpu --n_samples 3
# 한국어 베이스라인 재평가 (학습 불필요 — 무조건화 모델 재사용)
python training/evaluate_generation.py --lang ko \
  --ckpt checkpoints/round1_emotion_prefix/poem_ko_K0.pt --setup_id R3K_base \
  --eval_file data/processed/poem_ko_test_plain_r30_sowol.jsonl --use_gpu --n_samples 3

# 풀 지표 채점 (전 setup 반복; --lang en/ko)
python training/score_generations.py --gen_file results/gen_R3_orc_m.jsonl --lang en --use_gpu

# 집계 + 보고서
python analysis/analyze_round3.py
python analysis/make_report_round3.py
```

---

## 4. Task 1: 기본 3 Downstream Task

직접 구현한 GPT-2가 sanity check를 통과한 뒤, 세 task를 full fine-tuning(전 파라미터)과 last-linear-layer(backbone 동결, 분류층만)로 학습. 공통: epoch 10, 자체 AdamW, best-dev 저장, seed 11711.

| Task | 데이터셋 | 모드 | lr | batch | dev acc | macro-F1 |
|---|---|---|---|---|---|---|
| 감성 분류 | SST (5-class) | full | 1e-5 | 64 | **0.514** | 0.496 |
| | | last-linear | 1e-3 | 64 | 0.445 | 0.400 |
| 감성 분류 | CFIMDB (2-class) | full | 1e-5 | 8 | **0.971** | 0.971 |
| | | last-linear | 1e-3 | 8 | 0.857 | 0.856 |
| 패러프레이즈 | Quora (283k) | full | 1e-5 | 32 | **0.896** | 0.889 |
| 소네트 생성 | Shakespeare (131편 학습) | full | 1e-5 | 8 | dev chrF 27.7-27.8 (후속 정량화) | — |

핵심 관찰: ① 두 분류 task 모두 full fine-tuning이 우수(+0.069/+0.114) — 이후 모든 실험에서 full 고정의 근거. ② 예측 분포가 gold와 유사(다수클래스 붕괴 없음 — CFIMDB last-linear 초기 붕괴는 재학습으로 정상화). ③ 점수는 dev 기준(진짜 test는 정답 비공개). 상세: `docs/NLP_GPT2_실험_보고서.pdf`

---

## 5. 최종 실험: 감정 곡선 전체 제공 하의 시 생성

### 5.1 연구 질문과 설계 논리

> **시의 정답 감정 곡선 전체(미래 행 포함)를 학습·추론 양쪽에 줘도 continuation 생성이 좋아지지 않는가?**

선행 실험들(전역 감정 라벨, prefix 행 단위 감정 흐름 — `docs/`의 중간 보고서 2편)이 모두 무효과로 끝난 뒤, 남는 반론 두 가지("정보가 부족했다" / "표현·해상도가 부적절했다")를 정면으로 닫는 **상한(upper-bound) 실험**. 미래 행 감정의 제공은 의도적 oracle 설계로, 배포 가능한 설정이 아니라 "정답을 다 알 때의 최대 효과"를 잰다.

**데이터 무결성:** 사전 점검에서 영어 dev #138 = 비공개 test #155(동일 시) 중복을 발견 → **모든 영어 학습·평가에서 #138 제외 (평가 n=11)**.

### 5.2 실험 구성 (13 setup)

| 언어 | setup | 조건 | 존재 이유 |
|---|---|---|---|
| EN | base | 없음 | 기준점 |
| EN | oracle × {m,v,r} | 그 시의 정답 14행 곡선 (이산 m / 연속 v·r, 상호상관 0.28-0.61) | 주 실험 — 측정 체계 의존성까지 확인 |
| EN | average × {m,v,r} | train 130편 평균 곡선 | '전형적 소네트 곡선' prior만의 효과 |
| EN | shuffled × {m,v,r} | 다른 시의 정답 곡선을 순환 오배정 | ★핵심 통제 — 형식·분포는 실제, 내용만 오류 |
| KO | base | 없음 (기존 무조건화 모델 재평가) | 기준점 |
| KO | 전행 44종 / 6종 | 시의 모든 행에 세부/축약 감정 라벨열 (주석자 5인 다수결) | 최대 해상도 상한 / 해상도 가설 분리 |

한국어: train은 KPoEM 전체 368편, 평가는 김소월 11편(선행 실험에서 유일한 긍정 단서가 나온 가장 유리한 조건), 입력은 시 길이의 30% 행.

**주요 하이퍼파라미터** (전 setup 동일 — 비교의 전제): 학습 epoch 10 / batch 8 / lr 1e-5 / full fine-tuning / best dev-loss 저장(dev도 동일 조건화) / max_length EN 384·KO 768(곡선 블록 길이 반영). 디코딩 nucleus(temp 0.8, top_p 0.9) / 시당 3샘플 평균 / max_new_tokens 200 / KO 반복억제(no_repeat_ngram 3, rep_penalty 1.3). 지표: chrF, BERTScore(EN roberta-large/KO mBERT), Volta distance(생성·정답에 동일 측정기 적용). 전체 근거는 보고서 4장.

### 5.3 결과 요약

**영어 (n=11):** base가 chrF(27.69)·BERTScore(0.833) **모두 10개 setup 중 1위**. 정답 곡선은 m·v에서 11편 중 10편 chrF 악화(Δ-2.4), 셔플 곡선과 체계적으로 구별되지 않음(m: 셔플 우세 / v: 동률 / r: oracle 우세이나 최저점 셔플 대비). 평균 곡선만 Volta distance 개선(2.94-3.52 vs 4.58) — 선행 실험과 독립 재현된 유일한 효과(구조 prior).

**한국어 (김소월 11편):** chrF/BERTScore는 노이즈 수준(Δ+0.15~0.22 / +0.001~0.006). **역설** — 정답 감정열을 받은 두 모델 모두 base보다 감정 흐름 추종이 나쁨(traj_corr: base +0.156 vs 44종 -0.025 / 6종 +0.041). 44종→6종 축약도 효과를 열지 못함(해상도 가설 기각).

**결론:** GPT-2 small 규모에서 텍스트 prefix 형태의 감정 조건화는 — 정보의 형식·해상도·분량을 어떻게 주든 — 시 생성을 제어하지 못한다. 분석·논의·한계는 `docs/감정곡선_Round3_실험_보고서.pdf` 7-8장.

---

## 6. 보고서 및 결과물

| 문서 | 내용 |
|---|---|
| `docs/NLP_GPT2_실험_보고서.pdf` | Task 1 (기본 3 task) |
| `docs/감정곡선_Round3_실험_보고서.pdf` | **최종 실험** (설계 근거·하이퍼파라미터·paired 분석·논의) |
| `docs/감정기반_시생성_실험_보고서.pdf`, `docs/감정흐름_시생성_후속실험_보고서.pdf` | 중간 실험 기록 |
| `results/round3_analysis.json` | 최종 실험 집계 (전체표 + paired) |
| `results/gen_*.jsonl`, `results/scored_*.jsonl` | 시별 생성물·채점 |
