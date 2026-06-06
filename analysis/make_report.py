# -*- coding: utf-8 -*-
"""감정 정보 기반 GPT-2 시 생성 실험 보고서 PDF 생성 (Phase 6).

results/analysis.json 을 읽어 수치를 채우므로, 재실행 시 최신 결과가 반영된다.
"""
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

A = json.load(open("results/analysis.json", encoding="utf-8"))
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBd", "C:/Windows/Fonts/malgunbd.ttf"))
pdfmetrics.registerFontFamily("Malgun", normal="Malgun", bold="MalgunBd", italic="Malgun", boldItalic="MalgunBd")

ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    fn = kw.pop("fontName", "Malgun")
    return ParagraphStyle(name, parent=base, fontName=fn, **kw)


H1 = style("H1", fontName="MalgunBd", fontSize=19, leading=25, spaceAfter=6, textColor=colors.HexColor("#1a3c6e"))
SUB = style("SUB", fontSize=10.5, leading=15, textColor=colors.HexColor("#555555"), spaceAfter=2)
H2 = style("H2", fontName="MalgunBd", fontSize=14, leading=20, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a3c6e"))
H3 = style("H3", fontName="MalgunBd", fontSize=11.5, leading=16, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#2c5aa0"))
BODY = style("BODY", fontSize=10, leading=15, spaceAfter=5, alignment=TA_LEFT)
SMALL = style("SMALL", fontSize=8.5, leading=12, textColor=colors.HexColor("#444444"))
CELL = style("CELL", fontSize=9, leading=12)
CELLC = style("CELLC", fontSize=9, leading=12, alignment=TA_CENTER)
CELLCB = style("CELLCB", fontName="MalgunBd", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.white)

story = []


def P(t, s=BODY):
    story.append(Paragraph(t, s))


def gap(h=4):
    story.append(Spacer(1, h))


def make_table(data, col_widths, header=True, align_center_cols=None, highlight_rows=None):
    align_center_cols = align_center_cols or []
    highlight_rows = highlight_rows or []
    tbl = []
    for r, row in enumerate(data):
        nr = []
        for c, cell in enumerate(row):
            if r == 0 and header:
                nr.append(Paragraph(str(cell), CELLCB))
            else:
                nr.append(Paragraph(str(cell), CELLC if c in align_center_cols else CELL))
        tbl.append(nr)
    t = Table(tbl, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0")))
    for r in range(1, len(data)):
        if r in highlight_rows:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff2cc")))
        elif r % 2 == 0:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fc")))
    t.setStyle(TableStyle(cmds))
    return t


def bar_chart(cats, values, color, vmax):
    d = Drawing(440, 170)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 35, 25, 370, 125
    bc.data = [values]
    bc.categoryAxis.categoryNames = cats
    bc.categoryAxis.labels.fontName = "Malgun"
    bc.categoryAxis.labels.fontSize = 9
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = vmax
    bc.valueAxis.labels.fontName = "Malgun"
    bc.valueAxis.labels.fontSize = 8
    bc.bars[0].fillColor = color
    bc.barLabels.fontName = "MalgunBd"
    bc.barLabels.fontSize = 9
    bc.barLabelFormat = "%0.1f"
    bc.barLabels.nudge = 8
    d.add(bc)
    return d


# ===================== 표지 =====================
gap(8)
P("감정 정보를 활용한 GPT-2 기반 시 생성 실험", H1)
P("Emotion-Aware Poem Continuation with GPT-2 / KoGPT2", SUB)
gap(6)
P("CS224N 스타일 GPT-2 Final Project — 확장 실험 &nbsp;|&nbsp; 작성일: 2026-06-04", SMALL)
gap(8)
story.append(Table([[""]], colWidths=[170 * mm],
                   style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor("#2c5aa0"))])))
gap(8)
P("본 보고서는 <b>감정 분류로 학습된 정보가 시 생성에 도움이 되는가</b>를 검증한 실험을 정리한다. "
  "영어(GPT-2)와 한국어(KoGPT2) 두 언어에서, (A) 예측된 감정 라벨을 prompt에 붙이는 방식과 "
  "(B) 감정 분류로 먼저 fine-tuning한 backbone을 시 생성으로 이어받는 방식을 baseline과 비교하였다. "
  "데이터 전처리·모델 학습·생성 평가의 전 과정과 최종 결과(CHRF)를 포함한다.", BODY)

# ===================== 1. 연구 질문 =====================
P("1. 연구 질문 및 동기", H2)
P("시 생성은 문법뿐 아니라 앞부분의 정서·분위기를 이어받아야 하는 어려운 생성 과제이다. "
  "GPT-2가 인간처럼 감정을 이해하지는 못하지만, 감정 분류 task를 통해 감정 관련 표현 패턴을 학습할 수 있다. "
  "이 표현이 시 생성으로 전이되는지를 다음 질문으로 검증한다.", BODY)
data = [
    ["#", "연구 질문"],
    ["주질문", "감정 supervision은 GPT-2 계열 모델의 시 continuation 성능(CHRF)을 향상시키는가?"],
    ["Q1", "예측된 감정 라벨을 prompt prefix로 넣으면 성능이 향상되는가? (Pipeline A)"],
    ["Q2", "감정 분류 fine-tuning 후 시 생성 fine-tuning을 하면 향상되는가? (Pipeline B)"],
    ["Q3", "이 효과가 영어 GPT-2와 한국어 KoGPT2에서 비슷하게 나타나는가?"],
    ["Q4", "영어/한국어 감정·시 데이터는 분포·길이·tokenization에서 어떤 차이를 보이는가?"],
]
story.append(make_table(data, [22 * mm, 148 * mm], align_center_cols=[0]))

# ===================== 2. 데이터셋 =====================
P("2. 데이터셋 및 전처리", H2)
P("감정 분류용 데이터(대규모)와 시 생성용 데이터(소규모)의 두 축으로 구성된다. "
  "외부 데이터는 repo에 커밋하지 않고 전처리 스크립트로 재생성한다.", BODY)
ed = A["emotion_label_distribution"]
data = [
    ["용도", "데이터셋", "언어", "규모(train)", "형태"],
    ["감정 분류", "dair-ai/emotion", "영어", f"{ed['en']['n']:,}", "6-class single-label"],
    ["감정 분류", "KOTE → 6-class 변환", "한국어", f"{ed['ko']['n']:,}", "43-class multi → 6-class single"],
    ["시 생성", "Shakespeare Sonnets", "영어", "131편", "첫 3행 → 나머지"],
    ["시 생성", "KPoEM (근대시)", "한국어", "368편", "첫 3행 → 나머지"],
]
story.append(make_table(data, [24 * mm, 46 * mm, 20 * mm, 32 * mm, 48 * mm], align_center_cols=[2, 3]))
P("감정 6-class 공통 라벨: sadness, joy, love, anger, fear, surprise. KOTE는 rater 다수결(≥3) 기반 "
  "strict single-label subset으로 변환하였다. 시 데이터는 poem 단위 80/10/10 split(고정 seed).", SMALL)

P("2.1 감정 라벨 분포 (train)", H3)
P("두 언어의 분포가 크게 다르다(Q4). 영어는 joy, 한국어는 anger가 지배적이며, 이 불균형 때문에 "
  "분류 학습 시 class weight 보정이 필요했다.", BODY)
rows = [["언어"] + LABELS + ["불균형(최대/최소)"]]
for lang in ["en", "ko"]:
    c = ed[lang]["counts"]
    mx, mn = max(c.values()), max(1, min(c.values()))
    rows.append([lang.upper()] + [f"{c[l]}" for l in LABELS] + [f"{mx/mn:.1f}x"])
story.append(make_table(rows, [16 * mm] + [22 * mm] * 6 + [26 * mm], align_center_cols=list(range(1, 8))))

# ===================== 3. 모델 & 환경 =====================
P("3. 모델 및 실행 환경", H2)
P("영어/한국어는 코드 경로를 분리하였다(한국어 원본 세팅을 그대로 보존하기 위함). "
  "두 모델 모두 GPT-2 small 규모(hidden 768, 12 layers, 12 heads)로 비교 가능성을 맞췄다.", BODY)
data = [
    ["언어", "모델", "tokenizer", "비고"],
    ["영어", "직접 구현한 GPT2Model (gpt2 가중치 이식)", "gpt2 BPE (vocab 50,257)", "분류 head / tied LM head"],
    ["한국어", "skt/kogpt2-base-v2 (HuggingFace 원본)", "SentencePiece (vocab 51,200)", "AutoModelForCausalLM"],
]
story.append(make_table(data, [18 * mm, 62 * mm, 50 * mm, 40 * mm]))
P("실행 환경: conda <b>NLP_Project</b>, torch 2.11.0+cu128 (RTX 5070 Ti, 17GB), transformers 4.46.3, "
  "sacrebleu 2.5.1(정식 CHRF). KoGPT2 tokenizer는 special token을 명시적으로 지정해 로드해야 "
  "(pad=&lt;pad&gt;, eos=&lt;/s&gt;) 임베딩 범위 초과 오류를 피한다.", SMALL)

# ===================== 4. 실험 설계 =====================
P("4. 실험 설계", H2)
P("baseline(감정 미사용) 대비 두 가지 감정 주입 방식을 비교한다. 모든 setup은 동일한 학습 설정"
  "(epoch 10, batch 8, lr 1e-5, best dev-loss 저장)과 동일한 디코딩을 사용하여, 성능 차이가 오직 "
  "감정 정보에서 비롯되도록 통제하였다.", BODY)
data = [
    ["ID", "언어", "설정", "목적"],
    ["E0 / K0", "영/한", "시 SFT만 (baseline)", "감정 미사용 기준선"],
    ["E1 / K1", "영/한", "예측 감정 prefix + 시 SFT", "명시적 감정 conditioning (Pipeline A)"],
    ["E2 / K2", "영/한", "랜덤 감정 prefix + 시 SFT", "prefix 통제군 (감정 내용 효과 분리)"],
    ["E3 / K3", "영/한", "감정 분류 SFT → 시 SFT", "표현 전이 (Pipeline B)"],
]
story.append(make_table(data, [22 * mm, 18 * mm, 60 * mm, 70 * mm], align_center_cols=[1]))
P("E2/K2(랜덤 prefix)는 핵심 통제군이다. E1이 E0보다 좋아져도 그것이 '감정 정보' 때문인지 단순히 "
  "'prefix 토큰 추가' 때문인지 구분하려면 E1 vs E2 비교가 필요하다.", SMALL)
gap(2)
P("평가: 시의 첫 3행을 입력해 나머지를 생성하고 reference와 <b>CHRF</b>(character n-gram F-score)로 "
  "비교한다. 주지표는 continuation CHRF(생성된 뒷부분 vs 정답 뒷부분), 영어는 full-poem CHRF도 보고. "
  "디코딩은 nucleus sampling(temp 0.8, top_p 0.9), prompt당 3회 생성 후 CHRF 평균(소규모 평가셋의 "
  "분산 완화). 한국어는 반복 붕괴 방지를 위해 no_repeat_ngram=3, repetition_penalty=1.3을 "
  "모든 K setup에 동일 적용. 평가셋은 영어 12편, 한국어 47편.", BODY)

story.append(PageBreak())

# ===================== 5. 실험 과정 =====================
P("5. 실험 과정 (파이프라인)", H2)
data = [
    ["단계", "내용", "결과/비고"],
    ["Phase 0", "외부 데이터 다운로드·전처리, glue layer 검증", "CHRF identity 100.0 확인"],
    ["Phase 1", "6-class 감정 분류기 SFT (영/한)", "분류기 산출 (Pipeline A/B 전제)"],
    ["Phase 2", "시 생성 baseline SFT (E0/K0)", "EN 정상 수렴, KO epoch1 과적합"],
    ["Phase 3", "생성+CHRF 평가 인프라 구축", "한국어 반복붕괴 발견·해결"],
    ["Phase 4", "감정 prefix 실험 (E1/E2, K1/K2)", "Pipeline A"],
    ["Phase 5", "sequential SFT 실험 (E3/K3)", "Pipeline B"],
    ["Phase 6", "종합 분석 및 보고서", "본 문서"],
]
story.append(make_table(data, [20 * mm, 80 * mm, 70 * mm]))
P("Phase 3에서 KoGPT2가 plain sampling 시 같은 구절을 무한 반복(\"백골만 따라갔다\"×N)하여 CHRF가 "
  "바닥(~3)으로 떨어지는 문제를 발견하고, 반복 억제 디코딩으로 자연스러운 생성(CHRF ~9)을 회복하였다. "
  "한국어 시 데이터(368편)는 epoch 1 이후 dev loss가 상승하여, best dev-loss 체크포인트 저장으로 "
  "과적합을 자동 차단하였다.", SMALL)

# ===================== 6. 결과 =====================
P("6. 실험 결과", H2)

P("6.1 감정 분류기 성능 (Phase 1)", H3)
clf = A["classifier"]
rows = [["언어", "dev accuracy", "macro-F1"] + LABELS]
for lang in ["en", "ko"]:
    pcf = clf[lang]["per_class_f1"]
    rows.append([lang.upper(), f"{clf[lang]['dev_acc']:.3f}", f"{clf[lang]['dev_macro_f1']:.3f}"]
                + [f"{pcf[l]:.2f}" for l in LABELS])
story.append(make_table(rows, [14 * mm, 24 * mm, 20 * mm] + [18.7 * mm] * 6, align_center_cols=list(range(1, 9))))
P("영어가 한국어보다 높다(macro-F1 0.899 vs 0.704). 한국어 fear·surprise는 표본이 적어 약하지만, "
  "class weight 보정 덕분에 anger(48.6%)로의 다수클래스 붕괴는 발생하지 않아 다양한 라벨을 예측한다 "
  "— 이는 Pipeline A 실험이 성립하기 위한 전제이다.", SMALL)

P("6.2 시 생성 CHRF 결과 (전체 매트릭스)", H3)
m = A["chrf_matrix"]
rows = [["setup", "continuation CHRF", "full-poem CHRF", "baseline 대비"]]
order = ["E0", "E1", "E2", "E3", "K0", "K1", "K2", "K3"]
base = {"E": m["E0"]["continuation"], "K": m["K0"]["continuation"]}
hl = []
for idx, s in enumerate(order, start=1):
    b = base[s[0]]
    delta = m[s]["continuation"] - b
    dtxt = "— (기준)" if s in ("E0", "K0") else f"{delta:+.2f}"
    rows.append([s, f"{m[s]['continuation']:.2f}", f"{m[s]['full_poem']:.2f}", dtxt])
    if s in ("E0", "K0"):
        hl.append(idx)
story.append(make_table(rows, [24 * mm, 46 * mm, 46 * mm, 40 * mm], align_center_cols=[1, 2, 3], highlight_rows=hl))

gap(4)
en_vals = [m[s]["continuation"] for s in ["E0", "E1", "E2", "E3"]]
ko_vals = [m[s]["continuation"] for s in ["K0", "K1", "K2", "K3"]]
P("영어 (E0~E3) continuation CHRF", SMALL)
story.append(bar_chart(["E0", "E1", "E2", "E3"], en_vals, colors.HexColor("#2c5aa0"), 32))
P("한국어 (K0~K3) continuation CHRF", SMALL)
story.append(bar_chart(["K0", "K1", "K2", "K3"], ko_vals, colors.HexColor("#9e3b3b"), 10))

P("6.3 핵심 비교 (paired, 시별 대응 비교)", H3)
pr = A["paired"]
rows = [["비교", "평균 Δ", "향상/악화 (편수)", "해석"]]
mapping = [
    ("E1 vs E0", "E1_vs_E0", "예측 감정 prefix가 baseline을 이기는가?"),
    ("E1 vs E2", "E1_vs_E2", "예측 감정이 랜덤 prefix보다 나은가? (A의 핵심)"),
    ("E3 vs E0", "E3_vs_E0", "sequential SFT가 baseline을 이기는가?"),
]
for name, key, interp in mapping:
    p = pr[key]
    rows.append([name, f"{p['mean_delta']:+.2f}", f"{p['b_better']} / {p['b_worse']} (12편)", interp])
story.append(make_table(rows, [24 * mm, 22 * mm, 38 * mm, 86 * mm], align_center_cols=[1, 2]))
P("E1 vs E2가 +0.17(6:6 무승부)인 점이 결정적이다 — 올바르게 예측한 감정 라벨이 랜덤 라벨보다 나을 게 "
  "없으므로, 감정 내용 자체는 CHRF에 신호를 주지 못했다. E3는 11/12편에서 악화(평균 -4.14)되어 "
  "표현 전이가 오히려 시적 능력을 훼손했음을 보여준다.", SMALL)

P("6.4 추가 분석 — 길이 및 tokenization (Q4)", H3)
tl = A["tokenizer_length"]
rows = [["언어/tokenizer", "글자수/편", "토큰수/편", "글자/토큰"],
        ["영어 / gpt2", f"{tl['en_gpt2']['chars_per_poem_mean']:.0f}",
         f"{tl['en_gpt2']['tokens_per_poem_mean']:.0f}",
         f"{tl['en_gpt2']['chars_per_poem_mean']/tl['en_gpt2']['tokens_per_poem_mean']:.1f}"],
        ["한국어 / kogpt2", f"{tl['ko_kogpt2']['chars_per_poem_mean']:.0f}",
         f"{tl['ko_kogpt2']['tokens_per_poem_mean']:.0f}",
         f"{tl['ko_kogpt2']['chars_per_poem_mean']/tl['ko_kogpt2']['tokens_per_poem_mean']:.1f}"]]
story.append(make_table(rows, [44 * mm, 38 * mm, 38 * mm, 40 * mm], align_center_cols=[1, 2, 3]))
P("한국어 시는 글자수가 영어의 약 절반이지만 토큰수는 비슷하다 — 한국어 tokenization이 더 조밀하여 "
  "(글자당 토큰이 많아) 같은 편수라도 학습 신호의 성격이 다르다. 동일 epoch 비교가 불공정해질 수 있어 "
  "best dev-loss 기준 선택으로 통제하였다.", SMALL)

# ===================== 7. 결론 =====================
P("7. 결론 및 한계", H2)
P("<b>감정 supervision은 두 방법·두 언어 모두에서 시 continuation CHRF를 향상시키지 못했다.</b> "
  "Pipeline A에서는 예측 감정이 랜덤 prefix와 차이가 없었고(E1과 E2가 거의 동일), Pipeline B(sequential SFT)는 "
  "영어에서 오히려 성능을 일관되게 떨어뜨렸으며(11/12편 악화) 한국어는 변화가 없었다. "
  "이는 proposal의 Expected Outcome #3에 해당하는, 그 자체로 의미 있는 부정적 결과이다.", BODY)
P("원인으로는 (1) 감정 데이터(트윗·댓글)와 시(소네트·근대시)의 큰 도메인 불일치, "
  "(2) 시 데이터의 절대량 부족(131/368편), (3) CHRF가 reference 표면 유사도만 측정하여 정서적·시적 "
  "품질을 반영하지 못하는 점을 들 수 있다. 특히 (3) 때문에 감정적으로 잘 이어진 시도 reference와 다르면 "
  "낮은 CHRF를 받을 수 있다.", BODY)
P("향후 과제: 시 도메인에 더 가까운 감정 데이터, human preference 또는 LLM-as-a-judge 평가, "
  "생성 시의 emotion consistency·다양성(Distinct-n) 측정을 통해 CHRF의 한계를 보완할 수 있다.", BODY)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Malgun", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"- {doc.page} -")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, "감정 기반 GPT-2 시 생성 실험 보고서")
    canvas.restoreState()


doc = SimpleDocTemplate("docs/감정기반_시생성_실험_보고서.pdf", pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
                        title="감정 기반 GPT-2 시 생성 실험 보고서")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF 생성 완료: docs/감정기반_시생성_실험_보고서.pdf")
