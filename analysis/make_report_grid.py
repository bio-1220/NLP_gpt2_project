# -*- coding: utf-8 -*-
"""Round-2 진단 실험(감정 흐름 trajectory guidance) 보고서 PDF 생성.

results/grid_analysis.json 을 읽어 수치를 채운다.
주의: Malgun 폰트는 U+2212(−)·U+2248 글리프가 없으므로 ASCII '-'만 사용.
"""
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

A = json.load(open("results/grid_analysis.json", encoding="utf-8"))

pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBd", "C:/Windows/Fonts/malgunbd.ttf"))
pdfmetrics.registerFontFamily("Malgun", normal="Malgun", bold="MalgunBd", italic="Malgun", boldItalic="MalgunBd")
ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    fn = kw.pop("fontName", "Malgun")
    return ParagraphStyle(name, parent=base, fontName=fn, **kw)


H1 = style("H1", fontName="MalgunBd", fontSize=18, leading=24, spaceAfter=6, textColor=colors.HexColor("#1a3c6e"))
SUB = style("SUB", fontSize=10.5, leading=15, textColor=colors.HexColor("#555555"), spaceAfter=2)
H2 = style("H2", fontName="MalgunBd", fontSize=13.5, leading=19, spaceBefore=13, spaceAfter=5, textColor=colors.HexColor("#1a3c6e"))
H3 = style("H3", fontName="MalgunBd", fontSize=11, leading=15, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#2c5aa0"))
BODY = style("BODY", fontSize=9.8, leading=14.5, spaceAfter=5, alignment=TA_LEFT)
SMALL = style("SMALL", fontSize=8.3, leading=11.5, textColor=colors.HexColor("#444444"))
CELL = style("CELL", fontSize=8.6, leading=11.5)
CELLC = style("CELLC", fontSize=8.6, leading=11.5, alignment=TA_CENTER)
CELLCB = style("CELLCB", fontName="MalgunBd", fontSize=8.6, leading=11.5, alignment=TA_CENTER, textColor=colors.white)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))
gap = lambda h=4: story.append(Spacer(1, h))


def make_table(data, col_widths, align_center_cols=None, highlight_rows=None):
    align_center_cols = align_center_cols or []
    highlight_rows = highlight_rows or []
    tbl = []
    for r, row in enumerate(data):
        tbl.append([
            Paragraph(str(c), CELLCB if r == 0 else (CELLC if i in align_center_cols else CELL))
            for i, c in enumerate(row)
        ])
    t = Table(tbl, colWidths=col_widths, repeatRows=1)
    cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0"))]
    for r in range(1, len(data)):
        if r in highlight_rows:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff2cc")))
        elif r % 2 == 0:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fc")))
    t.setStyle(TableStyle(cmds))
    return t


def fv(v, nd=3):
    return "--" if v is None else f"{v:.{nd}f}"


METRIC_COLS = [("chrf", "CHRF", 2), ("bertscore_f1", "BERTScore", 3), ("traj_mse", "TrajMSE", 3),
               ("traj_corr", "TrajCorr", 3), ("volta_dist", "Volta", 2), ("line_emotion_agreement", "행감정일치", 3)]

NAME = {"K_base": "baseline", "K_orc": "oracle traj", "K_avg": "average traj", "K_rnd": "random traj",
        "E_base": "baseline", "E_orc": "Method A oracle", "E_avg": "Method B average",
        "S_base": "Sowol baseline", "S_orc": "Sowol oracle"}


def group_table(group_key, highlight=1):
    table = A["groups"][group_key]
    rows = [["setup"] + [c[1] for c in METRIC_COLS]]
    for s, m in table.items():
        rows.append([s.replace("_", " ")] + [fv(m.get(k), nd) for k, _, nd in METRIC_COLS])
    return make_table(rows, [30 * mm] + [23 * mm] * 6, align_center_cols=list(range(1, 7)), highlight_rows=[highlight])


# ===================== 표지 =====================
gap(8)
P("감정 흐름(Trajectory) 기반 시 생성 — 후속 진단 실험 보고서", H1)
P("Line-level Emotion / Sentiment Trajectory Guidance for GPT-2 Poem Continuation", SUB)
P("1차 실험 후속 | KPoEM line trajectory + Shakespeare Volta Sentiment | 작성일: 2026-06-06", SMALL)
gap(8)
story.append(Table([[""]], colWidths=[170 * mm], style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor("#2c5aa0"))])))
gap(8)
P("1차 실험에서 global emotion prefix와 sequential SFT는 시 continuation을 개선하지 못했다. 본 후속 실험은 "
  "실패 원인을 분해한다: <b>감정 라벨이 너무 거칠어서(coarse) 실패했는가? 행 단위로 구조화된 감정 흐름을 주면 "
  "달라지는가?</b> 이를 위해 (1) KPoEM line-level 감정 trajectory(한국어), (2) Shakespeare 14행 sentiment "
  "trajectory(영어), (3) prefix 비율 30/50/70%, (4) 김소월 시인 특화의 네 축으로 oracle / average / random "
  "통제 비교를 수행하고, CHRF 외에 BERTScore와 감정 궤적 지표(MSE·Correlation·Volta)로 평가하였다.", BODY)

# ===================== 1. 설계 =====================
P("1. 실험 설계", H2)
data = [
    ["축", "조건", "통제 논리"],
    ["KO 첫3행", "baseline / oracle / average / random trajectory", "oracle이 random보다 좋아야 '감정 내용' 신호"],
    ["KO ratio", "30% / 50% / 70% prefix x 위 4조건", "문맥(감정 흐름)을 많이 줄수록 효과가 커지는가"],
    ["EN sonnet", "baseline / Method A(oracle 14행 곡선) / Method B(train 평균 곡선)", "정답 감정 곡선의 upper-bound 효과"],
    ["김소월", "Sowol 전용 학습(131편) + 전체모델의 Sowol-test 교차평가", "감정 전개가 정형화된 시인에서의 효과"],
]
story.append(make_table(data, [22 * mm, 84 * mm, 64 * mm]))
P("공통: 학습은 trajectory prompt + 시 전체의 LM fine-tuning(conditioned_full_text), 평가는 model_input "
  "prompt로 생성 후 reference 비교. 미래 행의 감정은 입력에 쓰지 않으며(leakage-safe), 평균 곡선은 train에서만 "
  "계산. 모든 setup 동일 디코딩(temp 0.8, top_p 0.9, 시당 3샘플 평균; KO는 반복억제 동일 적용). "
  "epoch 10, batch 8, best dev-loss 저장. 평가셋: KO 47편, EN 12편, Sowol 11편.", SMALL)

P("2. 평가 지표 (full tier)", H2)
data = [
    ["지표", "측정 대상", "방향"],
    ["CHRF", "표면(문자 n-gram) 유사도", "높을수록"],
    ["BERTScore F1", "의미 유사도 (contextual embedding)", "높을수록"],
    ["Trajectory MSE / Corr", "행별 감정(valence) 궤적의 거리 / 모양 일치", "MSE 낮을수록 / Corr 높을수록"],
    ["Volta distance", "감정 전환점 위치 차이 (prefix+생성 전체 곡선)", "낮을수록"],
    ["행감정 일치(KO)", "행별 6-class 감정 라벨 일치율", "높을수록"],
    ["Distinct-2", "반복 없는 다양성", "높을수록"],
]
story.append(make_table(data, [40 * mm, 92 * mm, 38 * mm]))
P("측정기: EN = 사전학습 SST-2 DistilBERT(외부 모델, leakage 없음) 행별 score, KO = 1차에서 학습한 KOTE 감정 "
  "분류기 행별 적용 + valence 매핑(joy/love=+1, surprise=0, sadness/anger/fear=-1). 생성물과 reference에 "
  "같은 측정기를 적용해 측정기 편향을 상쇄.", SMALL)

story.append(PageBreak())

# ===================== 3. 결과 =====================
P("3. 결과", H2)

P("3.1 한국어 — 첫 3행 trajectory (HANDOFF 최우선 실험)", H3)
story.append(group_table("KO_first3"))
pr = {(p["a"], p["b"]): p for p in A["paired"]}
oc = pr[("K_orc", "K_base")]["metrics"]; orn = pr[("K_orc", "K_rnd")]["metrics"]
P(f"CHRF·BERTScore는 전부 flat. traj_corr은 oracle(+{oc['traj_corr']['mean_delta']:.3f} vs base)이 올라 보이지만 "
  f"<b>random({A['groups']['KO_first3']['K_rnd']['traj_corr']:.3f})이 oracle({A['groups']['KO_first3']['K_orc']['traj_corr']:.3f})보다 높다</b> "
  f"— oracle vs random 차이는 +{orn['traj_corr']['mean_delta']:.3f}(17/32)로 무의미. 행 단위로 구조화해도 "
  "감정 '내용'은 신호를 만들지 못했다.", SMALL)

P("3.2 한국어 — prefix ratio 30/50/70%", H3)
for rk, title in [("KO_r30", "ratio 30%"), ("KO_r50", "ratio 50%"), ("KO_r70", "ratio 70%")]:
    P(title, SMALL)
    story.append(group_table(rk))
    gap(2)
P("어떤 ratio에서도 oracle이 random을 일관되게 이기지 못한다 (r70: oracle corr 0.110 vs random 0.156). "
  "문맥을 더 줘도 결론은 같다. r70의 CHRF 하락(약 5.8)은 짧아진 target과 생성 길이 상한의 영향으로 모든 조건에 "
  "동일하게 작용한다.", SMALL)

P("3.3 영어 — sentiment trajectory (Method A/B)", H3)
story.append(group_table("EN_sent"))
eb = pr[("E_orc", "E_base")]["metrics"]
P(f"<b>Oracle 곡선이 오히려 일관되게 해를 끼쳤다</b>: paired에서 CHRF 악화 {eb['chrf']['a_worse']}/12 "
  f"(Δ{eb['chrf']['mean_delta']:+.2f}), BERTScore 악화 {eb['bertscore_f1']['a_worse']}/12. traj_corr도 개선 없음. "
  "유일한 예외는 Method B(average)의 Volta distance 개선(3.42 vs base 4.50) — 평균 곡선이 sonnet의 구조적 "
  "전환 위치에 약한 prior로 작동했을 가능성. 또한 conditioning이 영어 생성의 Distinct-2를 낮춰(0.85 -> 0.74) "
  "반복을 늘렸다.", SMALL)

P("3.4 김소월 — 유일한 긍정적 단서", H3)
sc = A["sowol_cross"]
rows = [["전체-KPoEM 모델 (Sowol test 11편 한정)", "TrajCorr", "TrajMSE", "행감정일치", "BERTScore"]]
for s in ["K_base", "K_rnd", "K_avg", "K_orc"]:
    v = sc[f"{s}_on_sowol_test"]
    rows.append([NAME.get(s, s), fv(v["traj_corr"]), fv(v["traj_mse"]), fv(v["line_emotion_agreement"]), fv(v["bertscore_f1"])])
story.append(make_table(rows, [62 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm], align_center_cols=[1, 2, 3, 4], highlight_rows=[4]))
P("김소월 시 11편으로 한정하면 <b>oracle이 모든 통제군을 이긴다</b>: traj_corr 0.199 (random 0.127, average "
  "0.010, baseline -0.173), 행감정 일치 0.357 최고, TrajMSE 1.356 최저. 다만 random도 크게 오르므로 상당 "
  "부분은 prefix 형식 효과이고, n=11이라 시사적 수준이다. 감정 전개가 정형화된(그리움-상실-체념) 시인에서는 "
  "감정 가이드가 약한 신호를 가질 수 있다는 유일한 단서. 한편 Sowol 전용 학습(S_base/S_orc, 131편)은 "
  "전체-KPoEM 학습보다 못했고 내부 비교도 무신호 — 데이터 축소 손실이 특화 이득을 압도한다.", SMALL)

P("4. 결론", H2)
P("1) Global prefix(1차)에 이어 <b>행 단위로 구조화된 line-level 감정 trajectory도 시 continuation을 개선하지 "
  "못했다</b> — 모든 지표에서 oracle과 random의 차이가 없다(감정 '내용' 무신호). "
  "2) prefix 비율을 30-70%로 바꿔 문맥을 늘려도 결론은 변하지 않는다. "
  "3) 영어에서는 정답 sentiment 곡선조차 표면·의미 지표를 일관되게 악화시켰다(12/12). "
  "4) baseline들의 traj_corr이 0 근처라는 사실 자체가 중요하다: GPT-2 small은 애초에 reference의 감정 흐름을 "
  "따라가지 못하며, prompt 조건화로는 이를 바꾸지 못했다. "
  "5) 결론: <b>감정 분류 능력은 곧바로 생성 제어 능력으로 이어지지 않는다.</b> 김소월 한정 oracle 우위(3.4)가 "
  "유일한 향후 과제 단서다.", BODY)
P("한계: KO trajectory 측정기는 댓글 도메인 분류기라 측정 노이즈가 있고, valence 매핑은 6-class를 3값으로 "
  "압축한다. 평가셋이 작아(12-47편) 개별 비교의 검정력이 낮으며, 생성 길이 상한이 긴 target(r70)의 CHRF를 "
  "구조적으로 낮춘다. Method C(predicted planner)는 scope 외로 남겼다.", SMALL)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Malgun", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"- {doc.page} -")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, "감정 흐름 기반 시 생성 후속 실험 보고서")
    canvas.restoreState()


doc = SimpleDocTemplate("docs/감정흐름_시생성_후속실험_보고서.pdf", pagesize=A4,
                        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="감정 흐름 기반 시 생성 후속 실험 보고서")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF 생성 완료: docs/감정흐름_시생성_후속실험_보고서.pdf")
