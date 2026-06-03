# -*- coding: utf-8 -*-
"""GPT-2 downstream-task 프로젝트 실험 보고서 PDF 생성 스크립트."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, ListFlowable, ListItem)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---- 한글 폰트 등록 ----
pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBd', 'C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('Malgun', normal='Malgun', bold='MalgunBd',
                              italic='Malgun', boldItalic='MalgunBd')

# ---- 스타일 ----
ss = getSampleStyleSheet()
def style(name, **kw):
    base = kw.pop('parent', ss['Normal'])
    fn = kw.pop('fontName', 'Malgun')
    return ParagraphStyle(name, parent=base, fontName=fn, **kw)

H1 = style('H1', fontName='MalgunBd', fontSize=20, leading=26, spaceAfter=6, textColor=colors.HexColor('#1a3c6e'))
SUB = style('SUB', fontSize=10.5, leading=15, textColor=colors.HexColor('#555555'), spaceAfter=2)
H2 = style('H2', fontName='MalgunBd', fontSize=14, leading=20, spaceBefore=14, spaceAfter=6,
           textColor=colors.HexColor('#1a3c6e'))
H3 = style('H3', fontName='MalgunBd', fontSize=11.5, leading=16, spaceBefore=8, spaceAfter=3,
           textColor=colors.HexColor('#2c5aa0'))
BODY = style('BODY', fontSize=10, leading=15, spaceAfter=5, alignment=TA_LEFT)
SMALL = style('SMALL', fontSize=8.5, leading=12, textColor=colors.HexColor('#444444'))
CELL = style('CELL', fontSize=9, leading=12)
CELLB = style('CELLB', fontName='MalgunBd', fontSize=9, leading=12)
CELLC = style('CELLC', fontSize=9, leading=12, alignment=TA_CENTER)
CELLCB = style('CELLCB', fontName='MalgunBd', fontSize=9, leading=12, alignment=TA_CENTER,
               textColor=colors.white)

story = []

def P(t, s=BODY): story.append(Paragraph(t, s))
def gap(h=4): story.append(Spacer(1, h))

def make_table(data, col_widths, header=True, align_center_cols=None, highlight_rows=None):
    align_center_cols = align_center_cols or []
    highlight_rows = highlight_rows or []
    tbl_data = []
    for r, row in enumerate(data):
        new_row = []
        for c, cell in enumerate(row):
            if r == 0 and header:
                new_row.append(Paragraph(str(cell), CELLCB))
            else:
                st = CELLC if c in align_center_cols else CELL
                new_row.append(Paragraph(str(cell), st))
        tbl_data.append(new_row)
    t = Table(tbl_data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    if header:
        cmds += [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0'))]
    for r in range(1, len(data)):
        if r in highlight_rows:
            cmds.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor('#eaf2fb')))
        elif r % 2 == 0:
            cmds.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor('#f7f9fc')))
    t.setStyle(TableStyle(cmds))
    return t

# ===================== 표지 =====================
gap(10)
P('GPT-2 Downstream Task 프로젝트', H1)
P('구현 · 환경 세팅 · 실험 및 결과 분석 보고서', SUB)
gap(6)
P('CS224N 스타일 GPT-2 Default Final Project &nbsp;|&nbsp; 작성일: 2026-06-03', SMALL)
gap(10)
story.append(Table([['']], colWidths=[170*mm], style=TableStyle(
    [('LINEABOVE', (0, 0), (-1, 0), 1.2, colors.HexColor('#2c5aa0'))])))
gap(8)
P('본 보고서는 직접 구현한 GPT-2 모델을 세 가지 downstream task '
  '(감성 분류 · 패러프레이즈 탐지 · 소네트 생성)에 적용하여 학습한 과정과 결과를 정리한 것이다. '
  '실행 환경 구축, 이식성 버그 수정, 두 가지 fine-tuning 모드(full-model / last-linear-layer) '
  '비교 실험까지 전 과정을 포함한다.', BODY)

# ===================== 1. 프로젝트 개요 =====================
P('1. 프로젝트 개요', H2)
P('GPT-2(124M, gpt2-small)를 처음부터 모듈 단위로 구현하고, HuggingFace 사전학습 가중치를 '
  '이식(remap)하여 세 가지 downstream task를 수행한다. 구현의 정확성은 <b>sanity_check.py</b>로 '
  '검증하였으며 "Your GPT2 implementation is correct!"를 통과하였다.', BODY)
data = [
    ['Task', '데이터셋', '형태', '예측 방식'],
    ['① 감성 분류', 'SST (5-class), CFIMDB (2-class)', '문장 분류', '마지막 토큰 → 분류 logit'],
    ['② 패러프레이즈 탐지', 'Quora Question Pairs', '문장쌍 분류', 'cloze 방식, "yes/no" 토큰 예측'],
    ['③ 소네트 생성', 'Shakespeare Sonnets (154편)', '언어 모델링', '전체 토큰 자기회귀 생성'],
]
story.append(make_table(data, [32*mm, 56*mm, 28*mm, 54*mm]))

# ===================== 2. 코드/파일 구조 =====================
P('2. 코드 및 파일 구조', H2)
P('핵심 구현 파일과 역할은 다음과 같다.', BODY)
data = [
    ['파일', '역할'],
    ['models/gpt2.py', 'GPT2Model 본체. 토큰/위치 임베딩, 트랜스포머 스택, 최종 LayerNorm, '
     '가중치 공유 LM 헤드(hidden_state_to_token), 사전학습 가중치 이식(from_pretrained).'],
    ['modules/attention.py', 'CausalSelfAttention. causal mask + padding mask를 적용한 멀티헤드 self-attention.'],
    ['modules/gpt2_layer.py', 'GPT2Layer. Pre-LayerNorm 구조의 트랜스포머 블록(attention→residual, FFN→residual).'],
    ['optimizer.py', 'AdamW 옵티마이저 구현.'],
    ['classifier.py', '감성 분류(SST·CFIMDB). full-model / last-linear-layer 두 모드 지원.'],
    ['paraphrase_detection.py', '패러프레이즈 탐지(Quora). ParaphraseGPT 모델 + 학습/평가.'],
    ['sonnet_generation.py', '소네트 생성. top-p(nucleus) 샘플링 + temperature 기반 생성.'],
    ['datasets.py', '데이터 로딩 및 collate. 각 task별 Dataset 클래스.'],
    ['evaluation.py', '평가 함수(accuracy, F1, paraphrase/sonnet 평가).'],
    ['sanity_check.py', 'GPT-2 구현 정확성 검증 스크립트.'],
]
story.append(make_table(data, [44*mm, 126*mm]))

# ===================== 3. 실행 환경 =====================
P('3. 실행 환경 세팅', H2)
P('GPU가 <b>RTX 5070 Ti (Blackwell, compute capability sm_120)</b>인데, 프로젝트의 env.yml이 설치하는 '
  '기본 torch는 CUDA 11.8(cu118) 빌드로 sm_120을 지원하지 않아 GPU 커널이 동작하지 않는 문제가 있었다. '
  '이를 해결하기 위해 <b>cu128 빌드의 최신 torch</b>로 새 conda 환경을 구성하였다.', BODY)
data = [
    ['항목', '내용'],
    ['conda 환경', 'NLP_Project (Python 3.10)'],
    ['PyTorch', 'torch 2.11.0+cu128 (RTX 5070 Ti sm_120 GPU 연산 정상 동작 확인)'],
    ['주요 의존성', 'transformers 4.46.3, tokenizers 0.20, einops 0.8.0, sacrebleu 2.5.1, '
     'scikit-learn, importlib-metadata'],
    ['GPU', 'NVIDIA GeForce RTX 5070 Ti (16GB)'],
    ['설치 방법', 'pip install torch ... --index-url https://download.pytorch.org/whl/cu128'],
]
story.append(make_table(data, [34*mm, 136*mm]))

# ===================== 4. 이식성/버그 수정 =====================
P('4. 적용한 코드 수정 사항', H2)
P('Windows(한글 cp949 로캘) + 최신 torch 2.11 환경에서 starter 코드가 실패하여 다음 수정을 적용했다. '
  '앞의 세 가지는 환경 이식성 문제이고, 마지막은 모델 설계 버그이다.', BODY)
data = [
    ['# ', '구분', '수정 내용 / 사유'],
    ['1', '인코딩', 'datasets.py의 CSV open()에 encoding="utf-8" 지정. (기본 cp949로 열려 '
     'Quora UTF-8 데이터 디코딩 실패 → UnicodeDecodeError)'],
    ['2', 'torch 직렬화', 'classifier.py · paraphrase_detection.py의 torch.load(..., weights_only=False). '
     '(torch 2.6+ 기본값 weights_only=True가 체크포인트의 SimpleNamespace를 언피클 불가)'],
    ['3', '표준출력', '실행 시 PYTHONUTF8=1 설정. (소네트의 em-dash(—) print가 cp949 콘솔에서 인코딩 실패)'],
    ['4', '설계 버그', 'ParaphraseGPT 헤드를 Linear(d, 2) → gpt.hidden_state_to_token으로 변경. '
     '(데이터셋 label은 토큰 id 8505/3919인데 2-class 출력과 불일치 → CUDA assertion 크래시. '
     '전체 vocab 위에서 예측하도록 수정)'],
]
story.append(make_table(data, [8*mm, 22*mm, 140*mm], align_center_cols=[0]))
P('수정 후 작은 배치 검증에서 logits 형태가 (B, 50257)로 정상 출력되고 cross_entropy/argmax가 '
  '동작함을 확인한 뒤 본 학습을 진행하였다.', SMALL)

story.append(PageBreak())

# ===================== 5. 실험 세팅 =====================
P('5. 실험 세팅', H2)
P('모든 학습은 공통적으로 epoch 10, AdamW를 사용하였다. learning rate는 학습 대상에 따라 달라진다: '
  '<b>전체 모델을 미세조정할 때는 1e-5</b>(사전학습 지식 보존), '
  '<b>frozen GPT의 마지막 분류층만 학습할 때는 1e-3</b>(새 헤드를 빠르게 학습). '
  'batch size는 GPU 메모리 한계 내에서 설정하였다.', BODY)
data = [
    ['실험', 'fine-tune 대상', 'lr', 'epoch', 'batch', 'best dev 저장'],
    ['감성: SST (full)', '전체 모델', '1e-5', '10', '64', 'O (dev acc)'],
    ['감성: CFIMDB (full)', '전체 모델', '1e-5', '10', '8', 'O (dev acc)'],
    ['감성: SST (last-linear)', '마지막 Linear층', '1e-3', '10', '64', 'O (dev acc)'],
    ['감성: CFIMDB (last-linear)', '마지막 Linear층', '1e-3', '10', '8', 'O (dev acc)'],
    ['패러프레이즈: Quora', '전체 모델', '1e-5', '10', '32', 'O (dev acc)'],
    ['소네트 생성', '전체 모델', '1e-5', '10', '8', '매 epoch 저장'],
]
story.append(make_table(data, [44*mm, 32*mm, 16*mm, 16*mm, 16*mm, 28*mm],
                        align_center_cols=[2, 3, 4, 5]))
P('Quora는 학습 데이터가 283,003개로 가장 커서, batch 8(약 4.6시간)보다 batch 32(약 2.5시간, '
  '메모리 4.9GB)로 설정해 시간을 단축하였다. CFIMDB는 문서가 길어 batch 8로 고정된다.', SMALL)

# ===================== 6. 실험 결과 =====================
P('6. 실험 결과 및 분석', H2)

P('6.1 종합 성능', H3)
data = [
    ['Task', '모드', 'dev accuracy', 'macro-F1'],
    ['SST (5-class)', 'full-model', '0.514', '0.496'],
    ['SST (5-class)', 'last-linear-layer', '0.445', '0.400'],
    ['CFIMDB (2-class)', 'full-model', '0.971', '0.971'],
    ['CFIMDB (2-class)', 'last-linear-layer', '0.857', '0.856'],
    ['Quora (paraphrase)', 'full fine-tune', '0.896', '0.889'],
]
story.append(make_table(data, [44*mm, 50*mm, 38*mm, 38*mm],
                        align_center_cols=[2, 3], highlight_rows=[1, 3, 5]))
P('예측 파일에서 직접 재계산한 정확도가 학습 로그와 정확히 일치함을 확인하여 결과의 유효성을 검증하였다. '
  '(파란 음영 = full-model)', SMALL)

P('6.2 Fine-tuning 모드 비교 (감성 분류)', H3)
P('이 실험의 핵심은 "전체 미세조정 vs 마지막 층만 학습(linear probe)"의 비교이다.', BODY)
data = [
    ['Task', 'last-linear-layer', 'full-model', 'full의 이득'],
    ['SST (5-class)', '0.445', '0.514', '+0.069'],
    ['CFIMDB (2-class)', '0.857', '0.971', '+0.114'],
]
story.append(make_table(data, [44*mm, 44*mm, 40*mm, 38*mm], align_center_cols=[1, 2, 3]))
P('두 task 모두 full fine-tuning이 우수하다. 다만 CFIMDB는 쉬운 2-class 문제라 frozen GPT의 '
  'linear probe(0.857)만으로도 상당히 높은 성능을 보인다. 이는 "task 난이도에 따라 전체 미세조정의 '
  '이득 폭이 달라진다"는 점을 보여준다.', BODY)

P('6.3 예측 분포 분석', H3)
P('예측 분포가 정답(gold) 분포와 유사할수록 특정 클래스로 쏠리지 않고 건강하게 학습된 것이다.', BODY)
data = [
    ['실험', '예측 분포', 'gold 분포', '비고'],
    ['SST full (5-class)', '0:115 / 1:320 / 2:127 / 3:338 / 4:201',
     '0:139 / 1:289 / 2:229 / 3:279 / 4:165', 'gold와 형태 유사, 중립(2) 과소예측'],
    ['CFIMDB full (2-class)', '0:124 / 1:121', '0:123 / 1:122', '거의 완벽히 균형'],
    ['Quora full', '0:24866 / 1:15563', '0:25537 / 1:14892', 'gold와 근접'],
    ['CFIMDB last-linear (수정 전)', '0:20 / 1:225', '0:123 / 1:122', '다수클래스 붕괴 (불량)'],
    ['CFIMDB last-linear (재실행)', '0:140 / 1:105', '0:123 / 1:122', '균형 회복 (정상)'],
]
story.append(make_table(data, [42*mm, 50*mm, 40*mm, 38*mm], highlight_rows=[5]))
P('당초 존재하던 last-linear-layer 예측 파일은 CFIMDB에서 거의 전부를 한 클래스로 찍는 붕괴 상태'
  '(acc 0.580)였으나, lr 1e-3로 충분히 재학습하자 균형 잡힌 분포와 acc 0.857로 정상화되었다.', SMALL)

P('6.4 학습 추이 (epoch별 dev accuracy)', H3)
data = [
    ['epoch', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    ['SST full', '.315', '.443', '.490', '.496', '.508', '.500', '.504', '.507', '.495', '.514'],
    ['CFIMDB full', '.951', '.963', '.963', '.971', '.967', '.963', '.967', '.967', '.951', '.959'],
    ['Quora full', '.853', '.864', '.872', '.879', '.880', '.892', '.889', '.896', '.895', '.896'],
    ['SST last-lin', '.408', '.415', '.353', '.433', '.424', '.428', '.438', '.445', '.425', '.421'],
    ['CFIMDB last-lin', '.739', '.833', '.678', '.739', '.657', '.833', '.829', '.751', '.857', '.780'],
]
w = [26*mm] + [14.2*mm]*10
story.append(make_table(data, w, align_center_cols=list(range(1, 11))))
P('full-model은 비교적 안정적으로 수렴하는 반면, last-linear-layer(frozen GPT)는 표현력 제약으로 '
  'epoch 간 진동이 크다. best dev 시점의 체크포인트를 저장하므로 최종 모델은 진동과 무관하게 '
  '최고 성능 가중치가 보존된다.', SMALL)

P('6.5 소네트 생성', H3)
P('held-out 12편(id 0~11)에 대해 앞 3줄을 prompt로 받아 나머지를 생성하였다. '
  'train loss는 epoch 0의 5.047에서 epoch 9의 4.065로 꾸준히 감소했다. '
  '생성문은 셰익스피어풍 어휘·운율을 모사하나 의미적 일관성은 제한적이며, 이는 학습 데이터가 '
  '154편으로 적고 모델이 gpt2-small인 점을 고려하면 예상되는 수준이다.', BODY)

P('6.6 산출물 파일', H3)
data = [
    ['종류', '경로'],
    ['감성 예측(full)', 'predictions/full-model-{sst,cfimdb}-{dev,test}-out.csv'],
    ['감성 예측(last-linear)', 'predictions/last-linear-layer-{sst,cfimdb}-{dev,test}-out.csv'],
    ['패러프레이즈 예측', 'predictions/para-{dev,test}-output.csv'],
    ['소네트 생성', 'predictions/generated_sonnets.txt'],
    ['체크포인트', 'sst-classifier.pt, cfimdb-classifier.pt, 10-1e-05-paraphrase.pt, '
     '{0..9}_10-1e-05-sonnet.pt, *-full-model.pt(백업)'],
    ['학습 로그', 'logs/{classifier,paraphrase,sonnet,lastlinear}_*.log'],
]
story.append(make_table(data, [40*mm, 130*mm]))

# ===================== 7. 결론 =====================
P('7. 결론', H2)
P('직접 구현한 GPT-2가 sanity check를 통과하였고, cu128 환경 구축과 이식성/설계 버그 수정을 거쳐 '
  '세 downstream task를 모두 정상 학습하였다. 최종 성능은 SST 0.514, CFIMDB 0.971, Quora 0.896이며, '
  '감성 분류에서 full fine-tuning이 last-linear-layer 대비 일관되게 우수함을 확인하였다. '
  '특히 잘못 학습되어 있던 CFIMDB linear-probe 결과(0.580 붕괴)를 재실행으로 0.857로 정상화하여 '
  '재현 가능한 비교 실험을 완성하였다.', BODY)

# ===================== 빌드 =====================
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Malgun', 8)
    canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f'- {doc.page} -')
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, 'GPT-2 Downstream Task 실험 보고서')
    canvas.restoreState()

doc = SimpleDocTemplate('NLP_GPT2_실험_보고서.pdf', pagesize=A4,
                        leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm, bottomMargin=20*mm,
                        title='GPT-2 Downstream Task 실험 보고서')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('PDF 생성 완료: NLP_GPT2_실험_보고서.pdf')
