# -*- coding: utf-8 -*-
"""Round 3 보고서 PDF (논문 스타일): 전체 감정 곡선 조건화 — 영어 m/v/r + 한국어 44/6class.

results/round3_analysis.json (EN) + results/scored_R3K_*.{jsonl,summary.json} (KO)을 읽어
수치를 채운다. 주의: Malgun 폰트는 U+2212/U+2248 글리프가 없으므로 ASCII '-'만 사용.
"""
import json
import statistics as st
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

EN = json.load(open("results/round3_analysis.json", encoding="utf-8"))


def ko_rows(s):
    return {r["id"]: r for r in (json.loads(l) for l in open(f"results/scored_{s}.jsonl", encoding="utf-8"))}


def ko_summ(s):
    return json.load(open(f"results/scored_{s}.summary.json", encoding="utf-8"))["metrics"]


KO_SETUPS = ["R3K_base", "R3K_fine", "R3K_six"]
KO = {s: ko_summ(s) for s in KO_SETUPS}
KO_R = {s: ko_rows(s) for s in KO_SETUPS}


def ko_paired(a, b, metric, higher=True):
    ra, rb = KO_R[a], KO_R[b]
    shared = sorted(set(ra) & set(rb))
    d = [(ra[k][metric] - rb[k][metric]) for k in shared
         if ra[k].get(metric) is not None and rb[k].get(metric) is not None]
    if not d:
        return None
    sign = 1 if higher else -1
    return {"mean": st.mean(d), "better": sum(1 for x in d if sign * x > 0), "n": len(d)}


pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBd", "C:/Windows/Fonts/malgunbd.ttf"))
pdfmetrics.registerFontFamily("Malgun", normal="Malgun", bold="MalgunBd", italic="Malgun", boldItalic="MalgunBd")
ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    fn = kw.pop("fontName", "Malgun")
    return ParagraphStyle(name, parent=base, fontName=fn, **kw)


H1 = style("H1", fontName="MalgunBd", fontSize=17, leading=23, spaceAfter=6, textColor=colors.HexColor("#1a3c6e"))
SUB = style("SUB", fontSize=10, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=2)
H2 = style("H2", fontName="MalgunBd", fontSize=13, leading=18, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1a3c6e"))
H3 = style("H3", fontName="MalgunBd", fontSize=11, leading=15, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#2c5aa0"))
BODY = style("BODY", fontSize=9.7, leading=14.3, spaceAfter=5, alignment=TA_LEFT)
SMALL = style("SMALL", fontSize=8.3, leading=11.5, textColor=colors.HexColor("#444444"))
CELL = style("CELL", fontSize=8.5, leading=11.3)
CELLC = style("CELLC", fontSize=8.5, leading=11.3, alignment=TA_CENTER)
CELLCB = style("CELLCB", fontName="MalgunBd", fontSize=8.5, leading=11.3, alignment=TA_CENTER, textColor=colors.white)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))
gap = lambda h=4: story.append(Spacer(1, h))


def make_table(data, col_widths, align_center_cols=None, highlight_rows=None):
    align_center_cols = align_center_cols or []
    highlight_rows = highlight_rows or []
    tbl = []
    for r, row in enumerate(data):
        tbl.append([Paragraph(str(c), CELLCB if r == 0 else (CELLC if i in align_center_cols else CELL))
                    for i, c in enumerate(row)])
    t = Table(tbl, colWidths=col_widths, repeatRows=1)
    cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5aa0"))]
    for r in range(1, len(data)):
        if r in highlight_rows:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff2cc")))
        elif r % 2 == 0:
            cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fc")))
    t.setStyle(TableStyle(cmds))
    return t


fv = lambda v, nd=3: "--" if v is None else f"{v:.{nd}f}"

# ===================== 표지/초록 =====================
gap(6)
P("감정 곡선 전체 제공 하에서의 시 생성 — Round 3 실험 보고서", H1)
P("Full-Trajectory Emotion Conditioning for GPT-2 Poem Continuation (EN sentiment m/v/r + KO 44/6-class)", SUB)
P("Round 1·2 후속 상한(upper-bound) 실험 | dev #138 제외 세팅 | 작성일: 2026-06-07", SMALL)
gap(6)
story.append(Table([[""]], colWidths=[174 * mm], style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.HexColor("#2c5aa0"))])))
gap(6)
P("<b>초록.</b> 앞선 두 라운드에서 전역 감정 라벨(R1)과 행 단위 감정 흐름(R2)은 시 continuation을 개선하지 "
  "못했다. 본 라운드는 남은 마지막 가능성, 즉 <b>정보량의 상한</b>을 검증한다: 시의 정답 감정 곡선 전체(미래 행 "
  "포함)를 학습과 추론 양쪽에 제공해도 생성이 좋아지지 않는가? 영어(셰익스피어 소네트)에서는 서로 다른 세 "
  "sentiment 측정 체계(m/v/r)의 14행 곡선을, 한국어(KPoEM-김소월)에서는 44종 세부감정과 6종 축약 감정의 "
  "전행(全行) 라벨열을 조건으로 부여하고, 무조건화·평균곡선·셔플곡선 통제군과 paired 비교하였다. 결과: "
  "<b>11개 조건화 모델 중 어느 것도 무조건화 베이스라인의 chrF/BERTScore를 의미 있게 넘지 못했고</b>, 정답 "
  "곡선은 셔플 곡선과 구별되지 않았으며(내용 무신호), 한국어에서는 정답 감정열을 받은 모델이 오히려 기준 모델보다 "
  "감정 흐름을 덜 따라가는 역설이 관측되었다. 재현된 유일한 효과는 train 평균 곡선의 감정 전환점(volta) 위치 "
  "개선으로, 이는 내용이 아닌 구조 prior 효과로 해석된다.", BODY)

# ===================== 1. 배경 =====================
P("1. 배경과 연구 질문", H2)
P("R1(전역 라벨 1개: 예측 = 무작위), R2(prefix 행별 흐름: oracle = random)의 연속 실패 후 남는 반론은 "
  "두 가지였다. (i) <i>정보가 부족했다</i> — prefix 행의 감정만 줬으니 시 전체의 감정 전개는 몰랐다. "
  "(ii) <i>표현이 부적절했다</i> — 라벨 해상도(개수)나 측정 체계가 문제였다. Round 3는 이 둘을 정면으로 "
  "닫는 상한 실험이다: <b>미래 행을 포함한 전체 곡선</b>을 (i)에, <b>3가지 sentiment 체계(영어)와 "
  "2가지 라벨 해상도(한국어)</b>를 (ii)에 대응시킨다. 정답 전체를 알려줘도 안 된다면, prompt 조건화 방식 "
  "자체의 상한이 막혀 있다는 결론이 가능하다.", BODY)
P("주의: 미래 행 감정의 제공은 R2에서 금지했던 leakage를 <b>의도적으로</b> 허용한 것이다. 이는 배포 가능한 "
  "설정이 아니라 '정답을 다 알 때의 최대 효과'를 재는 oracle 설계이며, 본 보고서의 모든 해석은 이를 전제한다.", SMALL)

# ===================== 2. 데이터 무결성 =====================
P("2. 데이터 무결성: dev #138 제외", H2)
P("실험 전 점검에서 영어 평가셋(dev 12편)의 소네트 #138과 채점 비공개 test셋의 '#155'가 동일한 시임을 "
  "확인했다(구두점 2자 차이, 전수 교차검사상 유일한 중복; 부수적으로 test '#156'은 실제 144번의 오번호임도 "
  "확인). dev로 모델을 선택하면 test 1편을 간접 최적화하게 되므로, <b>Round 3의 모든 영어 학습(dev 기반 "
  "모델 선택 포함)과 평가에서 #138을 제외했다(평가 n=11)</b>. 학습 데이터(소네트 1-131)에는 138이 없어 "
  "학습 자체는 영향이 없다.", BODY)

# ===================== 3. 실험 설계 =====================
P("3. 실험 설계", H2)
P("3.1 실험 구성 (학습 모델 12개 + 한국어 베이스라인 재평가 1 = 13개 setup)", H3)
data = [
    ["언어", "setup", "조건 내용", "존재 이유"],
    ["EN", "base", "조건 없음 (시 본문만 학습)", "기준점"],
    ["EN", "oracle x {m,v,r}", "그 시의 정답 14행 곡선 (연속값, 3가지 측정 체계)", "주 실험: 정답 곡선의 효과 상한"],
    ["EN", "average x {m,v,r}", "train 130편 평균 곡선을 전 시에 동일 부착", "시별 내용 없이 '전형적 소네트 곡선' prior만의 효과"],
    ["EN", "shuffled x {m,v,r}", "다른 시의 정답 곡선을 순환 오배정", "핵심 통제: 곡선의 형식·분포는 실제, 내용만 오류 - oracle이 이를 못 넘으면 내용 무신호"],
    ["KO", "base", "조건 없음 (기존 무조건화 모델 재평가)", "기준점"],
    ["KO", "fine (44종)", "전행 세부감정 라벨열 (주석자 5인 다수결)", "최대 해상도에서의 상한"],
    ["KO", "six (6종)", "동일 라벨열을 6-class로 축약 (poetic 매핑)", "해상도 가설(44종이 과세분?) 분리 검증"],
]
story.append(make_table(data, [12 * mm, 32 * mm, 70 * mm, 60 * mm]))
P("셔플 통제가 무작위 값 통제보다 강한 이유: 곡선의 값 분포·자기상관 등 통계적 성질을 전부 보존한 채 "
  "시-곡선 대응만 파괴하므로, '실제같은 곡선이 붙어 있는 효과'와 '맞는 곡선이 붙어 있는 효과'를 정확히 "
  "분리한다. 순환 오배정(i번 시가 i+1번 곡선)은 결정적이며 자기 곡선을 받는 시가 없음을 보장한다.", SMALL)

P("3.2 설계 선택의 근거", H3)
data = [
    ["선택", "내용", "근거"],
    ["EN 3개 측정 체계", "sentiment_m(이산 -1/0/1), v(연속, VADER류), r(연속, 신경망류)", "상호상관 0.28-0.61로 서로 다른 측정치 - 결과가 특정 측정기 의존인지 확인"],
    ["KO test = 김소월 11편", "train은 전체 368편, 평가만 김소월", "R2에서 유일한 긍정 단서가 김소월 한정 oracle 우위였음 - 가장 유리한 조건에서 상한 측정. 김소월 전용 '학습'은 R2에서 데이터 축소 손실로 역효과 확인됨"],
    ["KO 입력 = 30% 행", "고정 '첫 3행' 대신 시 길이의 30%", "한국시는 4-216행으로 길이 가변 - 비율 분할이 공정. 30%는 단시에서도 최소 1행 입력 보장(최소 1행, 최대 n-1행 클램프)"],
    ["행 라벨 = 5인 다수결", "주석자 5인의 라벨 인스턴스 전체 최빈값(동률은 사전순)", "행당 평균 18.5개 라벨 인스턴스를 단일 라벨로 - 결정적 규칙으로 재현성 확보"],
    ["조건 = 텍스트 블록", "[sentiment trajectory]/[감정흐름] 자연어 블록 prefix", "아키텍처 수정 없이 사전학습 어휘 지식 활용. 라벨은 숫자 ID가 아닌 단어('슬픔')로 - GPT-2가 의미를 이미 아는 토큰"],
]
story.append(make_table(data, [30 * mm, 60 * mm, 84 * mm]))

story.append(PageBreak())

# ===================== 4. 하이퍼파라미터 =====================
P("4. 하이퍼파라미터 (전체 명시 + 선정 근거)", H2)
P("4.1 학습", H3)
data = [
    ["항목", "값", "근거"],
    ["random seed", "11711 (전 단계 공통)", "스타터 기본값. split·초기화·디코딩까지 고정해 재현성 보장. 단일 시드는 한계(8절)"],
    ["optimizer / lr", "AdamW(자체 구현) / 1e-5", "full-model fine-tuning 표준값 - 더 크면 사전학습 지식 파괴"],
    ["fine-tune 범위", "전체 파라미터 (full)", "조건 블록 활용법을 backbone이 학습해야 함 - head-only로는 prompt 형식 학습 불가"],
    ["epochs / batch", "10 / 8", "전 라운드와 동일(비교 일관성). EN은 10내 단조 개선, KO는 조기 과적합을 best-dev 저장이 차단"],
    ["체크포인트 선택", "best dev-loss (dev도 동일 조건화)", "KO 시 데이터(368편)는 1-2 epoch에 과적합 반전 - 자동 가드. dev를 조건화하지 않으면 선택 기준이 조건화 모델에 불리해짐"],
    ["max_length (EN 조건화)", "384", "소네트 약 160토큰 + 14행 곡선 블록 약 150토큰"],
    ["max_length (KO 조건화)", "768", "전행 라벨 블록이 행수에 비례(최장 216행) - 256이면 본문 절단. 김소월 평가 프롬프트는 실측 최대 172토큰으로 안전"],
    ["학습 목표", "full-text LM (조건블록+시 전체에 CE, 패딩 -100)", "가장 단순한 조건화. 단, 블록 토큰에도 loss가 걸리므로 dev loss는 base와 비교 불가(6.4절)"],
]
story.append(make_table(data, [34 * mm, 50 * mm, 90 * mm]))

P("4.2 디코딩·평가 (13개 setup 전부 동일 - 비교의 전제)", H3)
data = [
    ["항목", "값", "근거"],
    ["sampling", "nucleus, temperature 0.8 / top_p 0.9", "스타터 기본 1.2는 과발산으로 비교 노이즈 증가 - 다양성·일관성 절충점"],
    ["샘플 수", "시당 3회 생성, 점수 평균", "평가셋이 작아(11편) 단일 표본 분산이 큼. 시드 = 기준 + 1000x샘플 + 시번호로 결정적"],
    ["max_new_tokens", "200", "EN 11행/KO 평균 continuation 커버. 전 setup 동일 상한(긴 시는 절단 - 8절)"],
    ["KO 반복 억제", "no_repeat_ngram 3 / repetition_penalty 1.3", "KoGPT2 반복 붕괴 방지(R2에서 확립). 모든 KO setup 동일 적용"],
    ["평가셋", "EN dev 11편(138 제외) / KO 김소월 test 11편", "paired 비교(같은 시에 전 setup 적용)로 소표본 보완"],
]
story.append(make_table(data, [34 * mm, 56 * mm, 84 * mm]))

P("4.3 평가 지표와 측정기", H3)
data = [
    ["지표", "정의", "방향 / 비고"],
    ["chrF", "문자 n-gram F-score, 생성 continuation vs 정답 continuation (sacrebleu)", "높을수록. 표면 유사도"],
    ["BERTScore F1", "contextual embedding 의미 유사도 (EN roberta-large / KO mBERT)", "높을수록. 표현이 달라도 의미 포착"],
    ["Volta distance", "감정 곡선에서 |s(i+1)-s(i)| 최대 지점(전환점) 위치 차이, prefix+생성 전체 곡선 기준", "낮을수록. 구조(전환 위치) 충실도"],
    ["보조: traj_corr / 행감정 일치", "행별 감정 궤적의 Pearson 상관 / 행 라벨 일치율(KO)", "감정 흐름 추종도"],
    ["측정기", "EN: 사전학습 SST-2 DistilBERT(외부, leakage 없음) / KO: 자체 KOTE 분류기 + valence 매핑", "생성물과 정답에 동일 측정기 적용 - 측정기 편향 상쇄"],
]
story.append(make_table(data, [34 * mm, 86 * mm, 54 * mm]))

story.append(PageBreak())

# ===================== 5. 결과: 영어 =====================
P("5. 결과 — 영어 (n=11, 시당 3샘플 평균)", H2)
P("5.1 전체 지표", H3)
T = EN["table"]
order = ["R3_base", "R3_orc_m", "R3_avg_m", "R3_shuf_m", "R3_orc_v", "R3_avg_v", "R3_shuf_v", "R3_orc_r", "R3_avg_r", "R3_shuf_r"]
rows = [["setup", "chrF", "BERTScore", "Volta(낮을수록)", "traj_corr", "Distinct-2"]]
for s in order:
    m = T[s]
    rows.append([s.replace("R3_", "").replace("_", " "), fv(m["chrf"], 2), fv(m["bertscore_f1"]),
                 fv(m["volta_dist"], 2), fv(m["traj_corr"]), fv(m["distinct2"])])
story.append(make_table(rows, [26 * mm, 24 * mm, 26 * mm, 30 * mm, 26 * mm, 26 * mm],
                        align_center_cols=[1, 2, 3, 4, 5], highlight_rows=[1]))
P("base가 chrF(27.69)와 BERTScore(0.833) <b>모두에서 10개 setup 중 1위</b>다. 정답 곡선(oracle)은 세 측정 "
  "체계 모두에서 base 아래(25.3/25.4/26.5)이고, 평균·셔플도 마찬가지다. 조건화는 또한 Distinct-2를 일관되게 "
  "낮춰(0.84에서 0.77-0.81) 반복을 늘렸다 - R2와 동일한 부작용.", BODY)

P("5.2 paired 분석 (시별 대응 비교)", H3)
pr = {(p["a"], p["b"]): p for p in EN["paired"]}
rows = [["비교", "chrF Δ (개선/11)", "BERTScore Δ (개선/11)", "Volta Δ (개선/11)"]]
for c in ["m", "v", "r"]:
    for b, bn in [("R3_base", "base"), (f"R3_shuf_{c}", f"shuf {c}"), (f"R3_avg_{c}", f"avg {c}")]:
        p = pr[(f"R3_orc_{c}", b)]["metrics"]
        rows.append([f"oracle {c} vs {bn}",
                     f"{p['chrf']['mean_delta']:+.2f} ({p['chrf']['a_better']}/11)",
                     f"{p['bertscore_f1']['mean_delta']:+.4f} ({p['bertscore_f1']['a_better']}/11)",
                     f"{p['volta_dist']['mean_delta']:+.2f} ({p['volta_dist']['a_better']}/11)"])
story.append(make_table(rows, [40 * mm, 42 * mm, 48 * mm, 42 * mm], align_center_cols=[1, 2, 3]))
P("<b>(a) oracle vs base:</b> m·v는 chrF에서 11편 중 10편 악화(각 Δ-2.40/-2.31), BERTScore도 9편 악화 - "
  "일관된 손상. r은 가장 완만하나(Δ-1.19) 여전히 음수. "
  "<b>(b) oracle vs shuffled(핵심):</b> m은 셔플이 우세(Δ-1.79), v는 동률권(Δ-0.76, 4:7), r만 oracle "
  "우세(Δ+2.63, 9/11). 단 shuf_r은 전 setup 최저점(23.87)으로, 이 우세는 'oracle r이 좋다'기보다 'shuf r이 "
  "유독 나빴다'는 해석이 자연스럽고, oracle r 자체가 base를 넘지 못하므로 가설을 구제하지 못한다. 세 체계에서 "
  "방향이 제각각(셔플 우세/동률/oracle 우세)이라는 사실 자체가 내용 신호의 부재(노이즈 지배)를 시사한다. "
  "<b>(c) 평균 곡선의 volta 효과:</b> avg_v 2.94, avg_m 3.52로 base(4.58) 대비 뚜렷하며, R2의 동일 관측"
  "(3.42)을 독립 재현한다. 시별 정답 곡선(oracle)은 이 효과조차 내지 못한다(4.27-4.76) - 즉 volta 개선은 "
  "'전형적 소네트 구조'라는 형식 prior의 효과이지 시별 감정 내용의 효과가 아니다.", BODY)

# ===================== 6. 결과: 한국어 =====================
P("6. 결과 — 한국어 (김소월 11편, 30% 입력)", H2)
P("6.1 전체 지표", H3)
rows = [["setup", "chrF", "BERTScore", "Volta(낮을수록)", "traj_corr", "행감정 일치"]]
NAME = {"R3K_base": "base (무조건화)", "R3K_fine": "전행 44종 라벨", "R3K_six": "전행 6종 라벨"}
for s in KO_SETUPS:
    m = KO[s]
    rows.append([NAME[s], fv(m["chrf"], 2), fv(m["bertscore_f1"]), fv(m["volta_dist"], 2),
                 fv(m["traj_corr"]), fv(m["line_emotion_agreement"])])
story.append(make_table(rows, [40 * mm, 24 * mm, 26 * mm, 30 * mm, 26 * mm, 26 * mm],
                        align_center_cols=[1, 2, 3, 4, 5], highlight_rows=[1]))

P("6.2 paired 분석", H3)
rows = [["비교", "chrF Δ (개선/11)", "BERTScore Δ (개선/11)", "Volta Δ (개선/11)", "traj_corr Δ", "행일치 Δ"]]
for a, b in [("R3K_fine", "R3K_base"), ("R3K_six", "R3K_base"), ("R3K_six", "R3K_fine")]:
    c1 = ko_paired(a, b, "chrf"); c2 = ko_paired(a, b, "bertscore_f1")
    c3 = ko_paired(a, b, "volta_dist", higher=False)
    c4 = ko_paired(a, b, "traj_corr"); c5 = ko_paired(a, b, "line_emotion_agreement")
    rows.append([f"{NAME[a].split(' ')[0] if 'base' in a else a.replace('R3K_','')} vs {b.replace('R3K_','')}",
                 f"{c1['mean']:+.2f} ({c1['better']}/{c1['n']})",
                 f"{c2['mean']:+.4f} ({c2['better']}/{c2['n']})",
                 f"{c3['mean']:+.2f} ({c3['better']}/{c3['n']})",
                 f"{c4['mean']:+.3f} ({c4['better']}/{c4['n']})",
                 f"{c5['mean']:+.3f} ({c5['better']}/{c5['n']})"])
story.append(make_table(rows, [26 * mm, 30 * mm, 34 * mm, 30 * mm, 27 * mm, 27 * mm], align_center_cols=[1, 2, 3, 4, 5]))
P("<b>(a) 텍스트 지표:</b> 44종은 chrF +0.22(8/11)·BERTScore 동률, 6종은 chrF +0.15(9/11)·BERTScore "
  "+0.006(7/11) - 우세 편수는 있으나 효과 크기가 측정 노이즈 수준이다. "
  "<b>(b) 역설:</b> 정답 감정열을 통째로 받은 두 모델 모두 base보다 감정 흐름 추종이 <b>나쁘다</b> "
  "(traj_corr: base +0.156 vs 44종 -0.025/6종 +0.041; 행일치: base 0.326 vs 0.248/0.287). 정답을 쥐여줘도 "
  "따라가지 못할 뿐 아니라 오히려 방해가 된다. "
  "<b>(c) 해상도: </b>6종이 44종보다 흐름 지표에서 덜 손상되고 BERTScore가 근소 우위(+0.005, 8/11)이나, "
  "volta는 44종만 개선(1.73 vs base 2.30). 어느 해상도도 효과를 열지 못한다 - '44종 과세분' 가설 기각.", BODY)

story.append(PageBreak())

# ===================== 7. 논의 =====================
P("7. 논의 — 왜 정답을 줘도 안 되는가", H2)
P("<b>(1) 모델은 곡선을 '읽지' 않고 '외운다'.</b> 조건화 모델들의 dev LM loss(EN 2.43-2.83, KO 2.92-3.30)는 "
  "base(4.07/4.85+)보다 훨씬 낮지만, 이는 정형화된 곡선 블록 토큰의 예측이 쉬워서이지 시 본문을 더 잘 "
  "모델링해서가 아니다(비교 불가 수치). cross-entropy는 블록을 복제하는 법은 가르치지만 블록의 의미를 본문 "
  "생성에 연결(grounding)하도록 강제하는 항이 없다. 한국어의 역설(정답을 받고도 흐름 추종이 나빠짐)은 이 "
  "해석의 직접 증거다: 블록은 학습된 '장식'이고, 그 처리에 쓰인 용량만큼 본문 모델링이 간섭받는다.", BODY)
P("<b>(2) 신호 대 표본의 구조적 불리함.</b> 조건-본문 결합을 배울 표본이 영어 130편, 한국어 368편뿐이다. "
  "124M급 모델이 이 표본으로 '곡선의 i번째 값이 i번째 행의 정서를 지시한다'는 대응을 귀납하기는 어렵고, "
  "셔플 통제가 oracle과 구별되지 않는 것이 그 결과다. 반면 평균 곡선의 volta 효과는 '모든 시에 같은 패턴'이라 "
  "표본 전체가 한 패턴 학습에 기여하므로 학습 가능했다고 설명된다 - 본 실험에서 유일하게 재현되는 효과가 "
  "하필 '내용 없는 prior'라는 사실은 이 표본-신호 논리와 정확히 부합한다.", BODY)
P("<b>(3) 측정의 한계가 가리는 부분.</b> chrF/BERTScore는 reference 유사도이므로, 감정 흐름을 잘 따른 "
  "'다른 좋은 시'는 보상받지 못한다. 다만 이 한계만으로는 결과를 설명할 수 없다 - 감정 흐름 직접 지표"
  "(traj_corr·행일치)에서도 조건화가 이기지 못했고 한국어에선 오히려 졌기 때문이다. 즉 '지표가 못 봐서'가 "
  "아니라 '실제로 흐름 제어가 일어나지 않아서'에 가깝다.", BODY)
P("<b>(4) 실무적 함의.</b> 정보량(전체 곡선)·해상도(44/6종)·측정 체계(m/v/r)·언어(영/한)를 모두 바꿔도 "
  "결론이 불변이므로, 남는 경로는 조건화 방식 자체의 변경이다: 블록 토큰 loss 마스킹(복제 학습 제거), 디코딩 "
  "시점 제어(분류기 유도 샘플링 등), 행 단위 인터리빙(라벨-행 교차 배치), 또는 더 큰 backbone. 이는 future "
  "work로 남긴다.", BODY)

# ===================== 8. 한계 =====================
P("8. 한계", H2)
P("(1) 평가셋이 작다(언어당 11편, paired로 보완했으나 검정력 낮음). (2) 단일 시드 단일 실행 - 학습 분산 "
  "미측정. (3) KO 흐름 측정기는 댓글 도메인 분류기로 측정 노이즈가 있으며 valence 매핑은 6class를 3값으로 "
  "압축한다. (4) max_new_tokens 200이 긴 target을 절단한다(전 setup 동일 적용으로 비교는 공정). "
  "(5) 김소월 11편은 R2에서 유리한 단서가 나온 시인으로의 선택이므로, 본 결과는 '가장 유리한 조건에서도 "
  "실패'로 읽는 것이 옳다. (6) 예측 planner(Method C)는 본 라운드에서도 범위 외.", BODY)

# ===================== 9. 결론 =====================
P("9. 결론", H2)
P("정답 감정 곡선 전체를 학습·추론 양쪽에 제공하는 상한 설정에서도, 세 가지 sentiment 측정 체계(영어)와 두 "
  "가지 라벨 해상도(한국어) 모두에서 시 continuation 품질은 무조건화 베이스라인을 넘지 못했다. 정답 곡선은 "
  "셔플 곡선과 체계적으로 구별되지 않았고, 한국어에서는 정답 감정열이 오히려 감정 흐름 추종을 떨어뜨렸다. "
  "R1-R3을 관통하는 결론은 다음과 같다: <b>GPT-2 small 규모에서 텍스트 prefix 형태의 감정 조건화는 - 정보를 "
  "어떤 형식·해상도·분량으로 주든 - 시 생성을 제어하지 못한다.</b> 유일하게 재현되는 효과는 train 평균 곡선에 "
  "의한 감정 전환점 위치 개선뿐이며, 이는 시별 감정 내용이 아니라 장르의 구조적 전형에서 오는 prior 효과다.", BODY)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Malgun", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"- {doc.page} -")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, "감정 곡선 전체 제공 시 생성 실험 (Round 3)")
    canvas.restoreState()


doc = SimpleDocTemplate("docs/감정곡선_Round3_실험_보고서.pdf", pagesize=A4,
                        leftMargin=17 * mm, rightMargin=17 * mm, topMargin=15 * mm, bottomMargin=18 * mm,
                        title="감정 곡선 전체 제공 시 생성 실험 보고서 (Round 3)")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF 생성 완료: docs/감정곡선_Round3_실험_보고서.pdf")
