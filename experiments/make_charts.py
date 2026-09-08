#!/usr/bin/env python3
"""Generate SVG charts for EXP-2026-01 from the final n=5,000 eval numbers."""
import pathlib

OUT = pathlib.Path("/Users/mostafaelkabir/Desktop/HaramBlur/Innovation-lab/experiments/assets")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTE, GRID = "#1a1d25", "#8899aa", "#dfe3ea"
GREEN, AMBER, RED, BLUE = "#27ae60", "#f39c12", "#e74c3c", "#4f8ef7"

# ---------------------------------------------------------------------------
# 1. Child-vs-Adult accuracy by true age (the teen valley)
# ---------------------------------------------------------------------------
age_buckets = [
    ("0",100.0,422),("5",99.8,446),("10",59.2,591),("15",51.3,706),
    ("20",85.9,405),("25",97.3,403),("30",98.2,221),("35",97.2,252),
    ("40",99.2,250),("45",100.0,270),("50",99.5,203),("55",98.7,223),
    ("60",99.4,174),("65",99.5,185),("70",99.2,125),("75",100.0,51),
    ("80",100.0,29),("85",100.0,9),
]
W,H = 940,440
x0,x1,y0,y1 = 64,904,70,370          # plot area
n=len(age_buckets); slot=(x1-x0)/n; bw=slot*0.72
def yy(p): return y1-(p/100)*(y1-y0)
def color(p): return RED if p<70 else (AMBER if p<90 else GREEN)

s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">Child-vs-Adult accuracy by TRUE age</text>')
s.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="13" fill="{MUTE}">"Correct" = VLM Child/Adult call (cutoff: under 13) matches GT. In the 10–19 dip the error is one-way: VLM says Child, GT says Adult. n=4,966</text>')
for g in (0,25,50,75,100):
    yg=yy(g)
    s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}" stroke-width="1"/>')
    s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
for i,(lab,p,nn) in enumerate(age_buckets):
    bx=x0+i*slot+(slot-bw)/2; by=yy(p); bh=y1-by
    s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{color(p)}"/>')
    s.append(f'<text x="{bx+bw/2:.1f}" y="{by-4:.1f}" text-anchor="middle" font-size="9.5" fill="{INK}">{p:.0f}</text>')
    s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+14:.1f}" text-anchor="middle" font-size="10" fill="{MUTE}">{lab}</text>')
s.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+34:.1f}" text-anchor="middle" font-size="12" fill="{INK}">true age (5-year buckets, start age shown)</text>')
# valley callout
vx=x0+2*slot
s.append(f'<rect x="{vx-4:.1f}" y="{yy(59.2)-2:.1f}" width="{2*slot:.1f}" height="{y1-yy(59.2)+2:.1f}" fill="none" stroke="{RED}" stroke-width="2" stroke-dasharray="4 3"/>')
s.append(f'<text x="{vx+slot:.1f}" y="{yy(59.2)-16:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="{RED}">teen valley (10–19)</text>')
# legend
lx=x0
for c,t in ((GREEN,"≥90%"),(AMBER,"70–90%"),(RED,"under 70%")):
    s.append(f'<rect x="{lx}" y="{H-24}" width="12" height="12" rx="2" fill="{c}"/>')
    s.append(f'<text x="{lx+16}" y="{H-14}" font-size="11" fill="{MUTE}">{t}</text>'); lx+=90
s.append('</svg>')
(OUT/"age_accuracy_by_gt_age.svg").write_text("\n".join(s))

# ---------------------------------------------------------------------------
# helper: confusion-matrix heatmap
# ---------------------------------------------------------------------------
def conf_svg(title, subtitle, rows, cols, M, fname, w=680):
    nR,nC=len(rows),len(cols)
    cell=88; lx=140; ty=96; H=ty+nR*cell+70
    def cellcolor(ri,ci,frac):
        base = GREEN if rows[ri]==cols[ci] else RED
        # opacity by row-fraction
        op = 0.12+0.83*frac
        return base, op
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{w}" height="{H}" fill="white"/>')
    s.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="{INK}">{title}</text>')
    s.append(f'<text x="{w/2}" y="50" text-anchor="middle" font-size="12.5" fill="{MUTE}">{subtitle}</text>')
    s.append(f'<text x="{lx+nC*cell/2}" y="80" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">VLM predicted →</text>')
    for ci,c in enumerate(cols):
        s.append(f'<text x="{lx+ci*cell+cell/2}" y="{ty-6}" text-anchor="middle" font-size="12" fill="{INK}">{c}</text>')
    for ri,r in enumerate(rows):
        rowtot=sum(M[ri]) or 1
        s.append(f'<text x="{lx-10}" y="{ty+ri*cell+cell/2+4}" text-anchor="end" font-size="12" fill="{INK}">{r}</text>')
        for ci in range(nC):
            frac=M[ri][ci]/rowtot
            base,op=cellcolor(ri,ci,frac)
            cx=lx+ci*cell; cy=ty+ri*cell
            s.append(f'<rect x="{cx}" y="{cy}" width="{cell-4}" height="{cell-4}" rx="4" fill="{base}" fill-opacity="{op:.2f}" stroke="{GRID}"/>')
            txt="#ffffff" if op>0.55 else INK
            s.append(f'<text x="{cx+(cell-4)/2}" y="{cy+(cell-4)/2-2}" text-anchor="middle" font-size="15" font-weight="700" fill="{txt}">{M[ri][ci]}</text>')
            s.append(f'<text x="{cx+(cell-4)/2}" y="{cy+(cell-4)/2+14}" text-anchor="middle" font-size="10.5" fill="{txt}">{100*frac:.0f}%</text>')
    s.append(f'<text x="14" y="{ty+nR*cell/2}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}" transform="rotate(-90 14 {ty+nR*cell/2})">Ground truth →</text>')
    s.append(f'<text x="{lx}" y="{ty+nR*cell+34}" font-size="11" fill="{MUTE}">Green = correct (diagonal) · Red = error · shade = share of that GT row · % is row-normalized</text>')
    s.append('</svg>')
    (OUT/fname).write_text("\n".join(s))

# 2. age 4-class confusion (child/teen/adult/senior)
age_rows=["child","teen","adult","senior"]
age_M=[[1194,4,1,0],[543,209,43,0],[124,363,2060,25],[2,0,234,164]]
conf_svg("Age confusion matrix (4-class)",
         "Teens are overwhelmingly predicted 'child' (68%); seniors often 'adult' (58%). Excludes abstentions. n=4,966",
         age_rows, age_rows, age_M, "age_confusion_matrix.svg", w=560)

# 3. gender confusion (Woman/Man adults, + abstain column)
gen_rows=["Woman","Man"]; gen_cols=["Woman","Man","(abstain)"]
gen_M=[[1516,6,424],[15,1551,288]]
# for coloring, treat abstain as neither correct nor error -> grey handled separately
def gender_svg():
    rows,cols,M=gen_rows,gen_cols,gen_M
    cell=92; lx=120; ty=96; nR,nC=2,3; w=lx+nC*cell+30; H=ty+nR*cell+70
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{w}" height="{H}" fill="white"/>')
    s.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="{INK}">Gender confusion matrix (adults)</text>')
    s.append(f'<text x="{w/2}" y="50" text-anchor="middle" font-size="12.5" fill="{MUTE}">Near-perfect: 21 wrong in 3,088 committed (99.3%). Abstains ~19%, shown separately.</text>')
    s.append(f'<text x="{lx+nC*cell/2}" y="80" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}">VLM predicted →</text>')
    for ci,c in enumerate(cols):
        s.append(f'<text x="{lx+ci*cell+cell/2}" y="{ty-6}" text-anchor="middle" font-size="12" fill="{INK}">{c}</text>')
    for ri,r in enumerate(rows):
        rowtot=sum(M[ri]) or 1
        s.append(f'<text x="{lx-10}" y="{ty+ri*cell+cell/2+4}" text-anchor="end" font-size="12" fill="{INK}">{r}</text>')
        for ci in range(nC):
            frac=M[ri][ci]/rowtot; cx=lx+ci*cell; cy=ty+ri*cell
            if cols[ci]=="(abstain)": base=MUTE
            elif rows[ri]==cols[ci]: base=GREEN
            else: base=RED
            op=0.12+0.83*frac
            s.append(f'<rect x="{cx}" y="{cy}" width="{cell-4}" height="{cell-4}" rx="4" fill="{base}" fill-opacity="{op:.2f}" stroke="{GRID}"/>')
            txt="#ffffff" if op>0.55 else INK
            s.append(f'<text x="{cx+(cell-4)/2}" y="{cy+(cell-4)/2-2}" text-anchor="middle" font-size="15" font-weight="700" fill="{txt}">{M[ri][ci]}</text>')
            s.append(f'<text x="{cx+(cell-4)/2}" y="{cy+(cell-4)/2+14}" text-anchor="middle" font-size="10.5" fill="{txt}">{100*frac:.0f}%</text>')
    s.append(f'<text x="14" y="{ty+nR*cell/2}" text-anchor="middle" font-size="12" font-weight="600" fill="{INK}" transform="rotate(-90 14 {ty+nR*cell/2})">Ground truth →</text>')
    s.append('</svg>')
    (OUT/"gender_confusion_matrix.svg").write_text("\n".join(s))
gender_svg()

# ---------------------------------------------------------------------------
# 4. Prediction vs GT by age group — makes the CUTOFF + error DIRECTION explicit
# ---------------------------------------------------------------------------
def pred_by_group():
    # (name, age label, correct class, VLM-child%, VLM-adult%, n)
    groups=[
        ("Children","age 0–12","Child",99.6,0.4,1199),
        ("Teens","age 13–17","Adult",68.3,31.7,795),
        ("Adults","age 18–64","Adult",4.8,95.2,2572),
        ("Seniors","age 65+","Adult",0.5,99.5,400),
    ]
    W,H=920,440
    bx0=310; bw=520; rh=54; gap=32; ty=118
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18" font-weight="700" fill="{INK}">What did the VLM predict for each TRUE age group?</text>')
    s.append(f'<text x="{W/2}" y="52" text-anchor="middle" font-size="12.5" fill="{MUTE}">Our cutoff: Child = under 13 (pre-puberty); everyone 13+ should be Adult.</text>')
    s.append(f'<text x="{W/2}" y="70" text-anchor="middle" font-size="12.5" fill="{MUTE}">Bar = how the VLM split that group.  Green = matched GT · Red = VLM wrong.</text>')
    for i,(name,age,correct,vc,va,n) in enumerate(groups):
        y=ty+i*(rh+gap)
        s.append(f'<text x="{bx0-16}" y="{y+rh/2-4}" text-anchor="end" font-size="14" font-weight="700" fill="{INK}">{name}</text>')
        s.append(f'<text x="{bx0-16}" y="{y+rh/2+14}" text-anchor="end" font-size="11" fill="{MUTE}">{age} · should be {correct}</text>')
        wc=bw*vc/100; wa=bw*va/100
        cchild = GREEN if correct=="Child" else RED
        cadult = GREEN if correct=="Adult" else RED
        s.append(f'<rect x="{bx0}" y="{y}" width="{wc:.1f}" height="{rh}" fill="{cchild}"/>')
        if vc>=9:
            s.append(f'<text x="{bx0+wc/2:.1f}" y="{y+rh/2-3}" text-anchor="middle" font-size="12" font-weight="700" fill="white">VLM: Child</text>')
            s.append(f'<text x="{bx0+wc/2:.1f}" y="{y+rh/2+13}" text-anchor="middle" font-size="11" fill="white">{vc:.0f}%</text>')
        s.append(f'<rect x="{bx0+wc:.1f}" y="{y}" width="{wa:.1f}" height="{rh}" fill="{cadult}"/>')
        if va>=9:
            s.append(f'<text x="{bx0+wc+wa/2:.1f}" y="{y+rh/2-3}" text-anchor="middle" font-size="12" font-weight="700" fill="white">VLM: Adult</text>')
            s.append(f'<text x="{bx0+wc+wa/2:.1f}" y="{y+rh/2+13}" text-anchor="middle" font-size="11" fill="white">{va:.0f}%</text>')
        s.append(f'<text x="{bx0+bw+10}" y="{y+rh/2+4}" font-size="10.5" fill="{MUTE}">n={n}</text>')
    yteen=ty+1*(rh+gap)
    s.append(f'<circle cx="{bx0+6}" cy="{yteen+rh+16}" r="4" fill="{RED}"/>')
    s.append(f'<text x="{bx0+16}" y="{yteen+rh+20}" font-size="11.5" font-weight="700" fill="{RED}">68% of teens (GT = Adult) were predicted Child — this is the main failure (your age-15 case).</text>')
    s.append('</svg>')
    (OUT/"prediction_by_gt_group.svg").write_text("\n".join(s))
pred_by_group()

PURPLE = "#8e44ad"

# ---------------------------------------------------------------------------
# 5. Gender accuracy vs coverage by true age
# ---------------------------------------------------------------------------
def gender_by_age():
    data=[(0,98.6,17.3),(5,98.5,30.5),(10,99.3,47.9),(15,99.5,84.3),(20,98.5,97.6),
          (25,98.5,97.8),(30,100.0,100.0),(35,99.6,98.8),(40,99.2,99.2),(45,100.0,99.6),
          (50,99.5,100.0),(55,100.0,100.0),(60,99.4,100.0),(65,99.5,99.5),(70,99.2,99.2),
          (75,100.0,100.0),(80,96.6,100.0),(85,100.0,100.0),(90,100.0,100.0)]
    W,H=980,470; x0,x1,y0,y1=66,936,86,384
    n=len(data); slot=(x1-x0)/n; bw=slot*0.55
    def yy(p): return y1-(p/100)*(y1-y0)
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">Gender: accuracy stays ~99% at every age</text>')
    s.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12.5" fill="{MUTE}">The model just answers less often on young kids (blue bars = how often it commits). It abstains — it does not get them wrong. n=5,000</text>')
    for g in (0,25,50,75,100):
        yg=yy(g); s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    for i,(a,acc,cov) in enumerate(data):
        bx=x0+i*slot+(slot-bw)/2; by=yy(cov)
        s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y1-by:.1f}" rx="2" fill="{BLUE}" fill-opacity="0.28"/>')
        s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+14:.1f}" text-anchor="middle" font-size="9.5" fill="{MUTE}">{a}</text>')
    pts=" ".join(f"{x0+i*slot+slot/2:.1f},{yy(acc):.1f}" for i,(a,acc,cov) in enumerate(data))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{GREEN}" stroke-width="3"/>')
    for i,(a,acc,cov) in enumerate(data):
        s.append(f'<circle cx="{x0+i*slot+slot/2:.1f}" cy="{yy(acc):.1f}" r="3.3" fill="{GREEN}"/>')
    s.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+34:.1f}" text-anchor="middle" font-size="12" fill="{INK}">true age (5-year buckets, start age shown)</text>')
    s.append(f'<circle cx="{x0+4}" cy="{H-22}" r="5" fill="{GREEN}"/><text x="{x0+16}" y="{H-18}" font-size="11" fill="{MUTE}">gender accuracy (when it commits)</text>')
    s.append(f'<rect x="{x0+270}" y="{H-27}" width="12" height="12" fill="{BLUE}" fill-opacity="0.28"/><text x="{x0+286}" y="{H-18}" font-size="11" fill="{MUTE}">coverage (how often it commits)</text>')
    s.append('</svg>')
    (OUT/"gender_accuracy_by_age.svg").write_text("\n".join(s))
gender_by_age()

# ---------------------------------------------------------------------------
# 6. What the VLM called each true age (detected vs actual) — stacked
# ---------------------------------------------------------------------------
def vlm_group_by_age():
    data=[(0,100,0,0,0),(5,100,0,0,0),(10,95,4,1,0),(15,48,41,9,0),(20,14,48,36,0),
          (25,3,13,83,0),(30,2,2,95,0),(35,3,0,96,0),(40,1,0,98,0),(45,0,0,99,1),
          (50,0,0,97,2),(55,1,0,97,2),(60,1,0,93,6),(65,1,0,74,26),(70,1,0,66,34),
          (75,0,0,24,76),(80,0,0,10,90),(85,0,0,11,89),(90,0,0,0,100)]
    cols=[("child",RED),("teen",AMBER),("adult",GREEN),("senior",PURPLE)]
    W,H=980,480; x0,x1,y0,y1=66,936,96,404
    n=len(data); slot=(x1-x0)/n; bw=slot*0.74
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">What the VLM called each TRUE age (detected vs actual)</text>')
    s.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12.5" fill="{MUTE}">Each bar = how the model labeled people of that true age. It only reliably says "adult" from ~25+; ages 15–24 are a mixed zone.</text>')
    for g in (0,25,50,75,100):
        yg=y1-(g/100)*(y1-y0); s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    for i,row in enumerate(data):
        a=row[0]; vals=row[1:]; bx=x0+i*slot+(slot-bw)/2; acc=0
        for (name,col),v in zip(cols,vals):
            if v<=0: continue
            hh=(v/100)*(y1-y0); by=y1-acc-hh
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{hh:.1f}" fill="{col}"/>')
            if v>=14: s.append(f'<text x="{bx+bw/2:.1f}" y="{by+hh/2+4:.1f}" text-anchor="middle" font-size="9" fill="white">{v}</text>')
            acc+=hh
        s.append(f'<text x="{bx+bw/2:.1f}" y="{y1+14:.1f}" text-anchor="middle" font-size="9.5" fill="{MUTE}">{a}</text>')
    s.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+34:.1f}" text-anchor="middle" font-size="12" fill="{INK}">true age (5-year buckets, start age shown)</text>')
    lx=x0
    for name,col in cols:
        s.append(f'<rect x="{lx}" y="{H-26}" width="12" height="12" rx="2" fill="{col}"/><text x="{lx+16}" y="{H-16}" font-size="11" fill="{MUTE}">VLM said {name}</text>'); lx+=155
    s.append('</svg>')
    (OUT/"age_detected_vs_actual.svg").write_text("\n".join(s))
vlm_group_by_age()

# ---------------------------------------------------------------------------
# 7. Threshold sweep — accuracy / catch-kids / catch-adults at each cutoff
# ---------------------------------------------------------------------------
def threshold_sweep():
    groups=[("under 10","child = 9 and under",81.7,99.9,77.8),
            ("puberty","child = 12 and under",86.4,99.6,82.2),
            ("minor","child = 17 and under",89.3,97.8,83.5)]
    series=[("overall accuracy",INK),("catch real kids",GREEN),("catch real adults",BLUE)]
    W,H=780,470; x0,x1,y0,y1=72,708,92,384
    ng=len(groups); gslot=(x1-x0)/ng; nb=3; bw=gslot*0.22
    def yy(p): return y1-(p/100)*(y1-y0)
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">Accuracy at each "child" cutoff</text>')
    s.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12" fill="{MUTE}">Only these 3 cutoffs are measurable (the model reports age in bands). Catching real kids stays ~98%+ everywhere.</text>')
    for g in (0,25,50,75,100):
        yg=yy(g); s.append(f'<line x1="{x0}" y1="{yg:.1f}" x2="{x1}" y2="{yg:.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{x0-8}" y="{yg+4:.1f}" text-anchor="end" font-size="11" fill="{MUTE}">{g}%</text>')
    for gi,(name,sub,acc,ck,ca) in enumerate(groups):
        vals=[acc,ck,ca]; gc=x0+gi*gslot+gslot/2; start=gc-(nb*bw+(nb-1)*7)/2
        for bi,((lab,col),v) in enumerate(zip(series,vals)):
            bx=start+bi*(bw+7); by=yy(v)
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{y1-by:.1f}" rx="2" fill="{col}"/>')
            s.append(f'<text x="{bx+bw/2:.1f}" y="{by-5:.1f}" text-anchor="middle" font-size="10.5" font-weight="700" fill="{INK}">{v:.0f}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+18:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="{INK}">{name}</text>')
        s.append(f'<text x="{gc:.1f}" y="{y1+34:.1f}" text-anchor="middle" font-size="10.5" fill="{MUTE}">{sub}</text>')
    lx=x0
    for lab,col in series:
        s.append(f'<rect x="{lx}" y="{H-24}" width="12" height="12" rx="2" fill="{col}"/><text x="{lx+16}" y="{H-14}" font-size="11" fill="{MUTE}">{lab}</text>'); lx+=185
    s.append('</svg>')
    (OUT/"child_threshold_sweep.svg").write_text("\n".join(s))
threshold_sweep()

for f in ("prediction_by_gt_group.svg","age_accuracy_by_gt_age.svg","age_confusion_matrix.svg",
          "gender_confusion_matrix.svg","gender_accuracy_by_age.svg","age_detected_vs_actual.svg",
          "child_threshold_sweep.svg"):
    print("wrote", OUT/f, (OUT/f).stat().st_size, "bytes")
