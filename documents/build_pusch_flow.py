import os
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PUSCH_ChEst_ChEq_workflow.drawio")

def q(s):
    return escape(s, {'"': "&quot;"})

STAGE = ("#dae8fc", "#6c8ebf")    # kernel / stage
TENSOR = ("#e1d5e7", "#9673a6")   # tensor
INPUT = ("#fff2cc", "#d6b656")    # external / static input
OPT = ("#f5f5f5", "#999999")      # optional / alternative
NOTE = ("#ffffff", "#bbbbbb")

class Page:
    def __init__(self, name, w=1800, h=1700):
        self.name, self.w, self.h, self.cells, self.n = name, w, h, [], 0

    def _id(self):
        self.n += 1
        return f"{self.name[:3]}{self.n}"

    def box(self, label, x, y, w, h, colors=STAGE, shape="rounded=1;", extra="", align="center"):
        i = self._id()
        style = (f"{shape}whiteSpace=wrap;html=1;fillColor={colors[0]};strokeColor={colors[1]};"
                 f"fontSize=11;align={align};spacingLeft=6;spacingRight=6;{extra}")
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def tensor(self, label, x, y, w=300, h=56, colors=TENSOR):
        return self.box(label, x, y, w, h, colors, shape="shape=parallelogram;perimeter=parallelogramPerimeter;size=0.06;fixedSize=0;")

    def group(self, label, x, y, w, h, fill="#eef4fb"):
        i = self._id()
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#6c8ebf;dashed=1;"
                 f"verticalAlign=top;align=left;spacingLeft=10;spacingTop=4;fontStyle=1;fontSize=12;")
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(self, s, t, label="", dashed=False, color="#333333", exit=None, entry=None):
        i = self._id()
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;fontSize=10;"
                 f"strokeColor={color};labelBackgroundColor=#ffffff;" + ("dashed=1;" if dashed else ""))
        if exit:
            style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" edge="1" parent="1" source="{s}" target="{t}">'
                          f'<mxGeometry relative="1" as="geometry"/></mxCell>')
        return i

    def text(self, label, x, y, w, h, size=11, bold=False):
        i = self._id()
        style = f"text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize={size};" + ("fontStyle=1;" if bold else "")
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def xml(self):
        return (f'<diagram id="{self.name}" name="{self.name}"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" '
                f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" '
                f'pageHeight="{self.h}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                + "".join(self.cells) + "</root></mxGraphModel></diagram>")

def legend(p, x, y):
    p.text("<b>Legend</b>", x, y, 200, 20)
    p.box("Kernel / stage", x, y + 25, 150, 30, STAGE)
    p.tensor("Tensor [dims]", x, y + 65, 150, 30)
    p.box("Static / external input", x, y + 105, 150, 30, INPUT)
    p.box("Alternative / optional", x, y + 145, 150, 30, OPT)
    p.text("Dims: d0 = fastest (column-major).<br>NF = 12·nPrb, NH = dmrsAddlnPos+1,<br>Nd = data symbols. Shapes are per UE group except DataRx (per cell).",
           x, y + 185, 300, 60, 10)

# ============================== PAGE 1: PUSCH pipeline ==============================
p = Page("PUSCH ChEst-ChEq", 1900, 1900)
p.text("PUSCH receiver front end: channel estimation → equalization → soft demapping (per slot, per UE group)", 40, 10, 1400, 30, 18, True)
p.text("Launch order from PuschRx::fullSlotKernelLaunch() (cuPHY/src/cuphy_channels/pusch_rx.cpp). Steps 2–4 produce inputs for the equalizer.", 40, 42, 1400, 20)

XI, XS, XT = 40, 420, 1010        # inputs, stages, tensors
SW, TW = 520, 330

# Inputs
dataRx = p.tensor("<b>DataRx</b> (cell slot buffer)<br>C16F [NF_cell, 14, nRxAnt]", XI, 140, 320, 60, INPUT)
filt = p.box("<b>Static filters</b><br>pWFreq / pWFreq4 / pWFreqSmall (rank 3)<br>pShiftSeq / pUnShiftSeq (rank 2 / 1)<br>RKHS params", XI, 230, 320, 80, INPUT)
ueg = p.box("<b>UE-group params</b> (cuphyPuschRxUeGrpPrms_t)<br>DMRS positions, ports, scrambling ID,<br>nPrb, startPrb, QAM per layer, algo flags", XI, 330, 320, 80, INPUT)

# ---- Step 1: channel estimation
g1 = p.group("Step 1 — Channel estimation (ch_est/)   ×NH time instances   selected by chEstAlgo", XS - 20, 100, SW + 40, 560)
s1a = p.box("<b>0 · Legacy MMSE</b><br>windowedChEst[NoDftSOfdm]Kernel (nPrb ≥ 4)<br>smallChEst[NoDftSOfdm]Kernel (nPrb 1–3)", XS, 135, SW, 64)
s1b1 = p.box("<b>1 · Multi-stage MMSE + delay est.</b>  ① windowedChEstPreNoDftSOfdmKernel<br>LS on DMRS tones, accumulate delay statistics", XS, 215, SW, 56)
s1b2 = p.box("② chEstFilter[Prg]NoDftSOfdmDispatchKernel<br>delay compensation + MMSE filtering", XS, 290, SW, 50)
s1c = p.box("<b>2 · RKHS</b>  puschRkhsChEstKernel&lt;nZpDmrsSc&gt;<br>(HEst C32F, DataRx C16F, nRxAnt ≤ 4)", XS, 360, SW, 50)
s1d = p.box("<b>3 · LS only</b>  windowedChEstPreNoDftSOfdmKernel (LS estimates only)", XS, 430, SW, 40)
s1e1 = p.box("<b>TensorRT estimator</b> (chest factory)  prepareChestMlInputsKernel", XS, 490, SW, 40, OPT)
s1e2 = p.box("TensorRT engine → extractChestMlOutputsKernel", XS, 545, SW, 40, OPT)
p.text("Kernel tiling: nPrb ≥ 8 & multiple of 4 → 8 PRB in / 4 out per CTA; other nPrb ≥ 4 → 4 in / 2 out; nPrb 1–3 → small kernel. "
       "NoDftSOfdm variants when DFT-s-OFDM is off.", XS, 600, SW, 50, 10)
p.edge(s1b1, s1b2)
p.edge(s1e1, s1e2)
p.edge(s1d, s1e1, dashed=True, color="#999999")

lsEst = p.tensor("<b>DmrsLSEst</b><br>C32F [NF/2, nLayers, nRxAnt, NH]", XT, 200, TW)
acc = p.tensor("<b>DmrsAccum</b> C32F [2]   <b>DmrsDelayMean</b> R32F [NH]", XT, 290, TW)
hEst = p.tensor("<b>HEst</b><br>C32F [nRxAnt, nLayers, NF, NH]", XT, 420, TW, 64)

p.edge(dataRx, g1, exit=(1, 0.5), entry=(0, 0.1))
p.edge(filt, g1, exit=(1, 0.5), entry=(0, 0.25))
p.edge(ueg, g1, exit=(1, 0.5), entry=(0, 0.42))
p.edge(s1b1, lsEst, "write")
p.edge(s1b1, acc, "write", exit=(1, 0.8))
p.edge(lsEst, s1b2, "read", dashed=True, exit=(0.5, 1), entry=(1, 0.3))
p.edge(s1b2, hEst, "write", exit=(1, 0.6), entry=(0, 0.2))
p.edge(s1a, hEst, "write", exit=(1, 0.5), entry=(0, 0.05))
p.edge(s1c, hEst, "write", exit=(1, 0.5), entry=(0, 0.5))
p.edge(s1e2, hEst, "write", exit=(1, 0.5), entry=(0, 0.9))

# ---- Step 2: noise / interference
y2 = 700
s2 = p.box("<b>Step 2 — Noise / interference estimation</b> (pusch_noise_intf_est/)<br>noiseIntfEst[NoDftSOfdm]Kernel<br><i>if SINR enabled or MMSE-type equalizer</i>", XS, y2, SW, 70)
nv = p.tensor("<b>NoiseVarPreEq</b> R32F [nUes]", XT, y2 - 10, TW, 44)
lw = p.tensor("<b>LwInv</b> (inv. Cholesky of noise cov.)<br>C32F [nRxAnt, nRxAnt, nPrb]", XT, y2 + 45, TW, 52)
p.edge(s2, nv, "write", exit=(1, 0.3))
p.edge(s2, lw, "write", exit=(1, 0.8))
d2 = p.tensor("DataRx (DMRS symbols)", XI, y2 + 10, 320, 44, INPUT)
p.edge(d2, s2)
p.edge(hEst, s2, "read", dashed=True, exit=(0.5, 1), entry=(0.85, 0))

# ---- Step 3 (optional) + Step 4: CFO/TA
y4 = 830
s3 = p.box("<b>Step 3 (opt.) — RSRP</b> (pusch_rssi/) → Rsrp R32F [nUes]", XS, y4, SW, 36, OPT)
s4 = p.box("<b>Step 4 — CFO / timing-offset estimation</b> (cfo_ta_est/)<br>cfoTaEstLowMimoKernel", XS, y4 + 55, SW, 56)
cfo = p.tensor("<b>CfoEst</b> C32F [14, nUes]<br><b>CfoHz, TaEst</b> R32F [nUes]", XT, y4 + 20, TW, 52)
rot = p.tensor("<b>CfoPhaseRot</b> C32F [4, 8, maxUeGrps]<br><b>TaPhaseRot</b> C32F [8, nUeGrps]", XT, y4 + 85, TW, 52)
p.edge(s3, s4)
p.edge(s4, cfo, "write", exit=(1, 0.3))
p.edge(s4, rot, "write", exit=(1, 0.8))

# ---- Step 5: coefficients
y5 = 1000
g5 = p.group("Step 5 — Equalizer coefficients (channel_eq/)   ×NH time instances", XS - 20, y5, SW + 40, 190)
s5 = p.box("<b>eqCoeffAlgo</b>: 0 RZF · 1 MMSE (diag. noise) · 2 MMSE-IRC · 3 IRC + RBLW shrink · 4 IRC + OAS shrink<br>"
           "<b>Kernel by antennas:</b><br>nRxAnt 1–8 → eqMmseCoefCompLowMimoKernel<br>nRxAnt 16 → eqMmseCoefCompHighMimoKernel (v1/v2/v3)<br>"
           "nRxAnt 64, 16 layers → eqMmseCoefCompMassiveMimoKernel_v1", XS, y5 + 35, SW, 140, align="left")
coef = p.tensor("<b>EqCoef</b><br>C32F [nRxAnt, 12, nLayers, nPrb, NH]", XT, y5 + 30, TW, 56)
ree = p.tensor("<b>ReeDiagInv</b> (1 / post-eq error var.)<br>R32F [12, nLayers, nPrb, nTimeChEq]", XT, y5 + 110, TW, 56)
p.edge(s5, coef, "write", exit=(1, 0.3))
p.edge(s5, ree, "write", exit=(1, 0.8))
p.edge(hEst, g5, "read", dashed=True, exit=(1, 0.5), entry=(1, 0.15))
p.edge(nv, g5, "read", dashed=True, exit=(1, 0.5), entry=(1, 0.25))
p.edge(lw, g5, "read (IRC)", dashed=True, exit=(1, 0.5), entry=(1, 0.35))
p.edge(s4, g5)

# ---- Step 6: equalize + soft demap
y6 = 1240
g6 = p.group("Step 6 — Equalization + soft demapping (channel_eq/)", XS - 20, y6, SW + 40, 175)
s6 = p.box("eqMmseSoftDemapKernel (≤ 16 Rx ant)  ·  eqMmseSoftDemapKernel_64R (64 Rx ant)<br>"
           "<i>variants:</i> eqMmseEqualizationKernel (equalize only, no soft demap)<br>eqMmseSoftDemap72eKernel (O-RAN split 7.2e)<br>"
           "symbol bitmask: full slot 0x3FFF · early-HARQ 0xF / 0x1F", XS, y6 + 35, SW, 120, align="left")
d6 = p.tensor("DataRx (data symbols)", XI, y6 + 60, 320, 44, INPUT)
llr = p.tensor("<b>LLR</b> (and LLRCdm1)<br>R16F [8, nLayers, NF, Nd]", XT, y6 + 10, TW, 56)
deq = p.tensor("<b>DataEq</b> (debug / eq-only / 7.2e)<br>C16F [nLayers, NF, Nd]", XT, y6 + 80, TW, 52, OPT)
dft = p.tensor("<b>DataEqDft</b> (DFT-s-OFDM UEs)<br>C32F [NF·Nd]", XT, y6 + 150, TW, 52)
p.edge(d6, g6)
p.edge(g5, g6)
p.edge(coef, g6, "read", dashed=True, exit=(1, 0.5), entry=(1, 0.2))
p.edge(ree, g6, "read", dashed=True, exit=(1, 0.5), entry=(1, 0.3))
p.edge(rot, g6, "read (CFO corr.)", dashed=True, exit=(1, 0.5), entry=(1, 0.4))
p.edge(s6, llr, "write", exit=(1, 0.2))
p.edge(s6, deq, "write", exit=(1, 0.6))
p.edge(s6, dft, "write (TP UEs)", exit=(1, 0.95))

# ---- Step 7: DFT-s-OFDM
y7 = 1470
g7 = p.group("Step 7 — DFT-s-OFDM only (enableDftSOfdm = 1)", XS - 20, y7, SW + 40, 160, "#f7f7f7")
s7a = p.box("7a · bluestein_Idft_kernel<br>IDFT of size NF per data symbol, in place on DataEqDft", XS, y7 + 35, SW, 48)
s7b = p.box("7b · eqMmseSoftDemapAfterDftKernel", XS, y7 + 100, SW, 40)
ws = p.box("Bluestein workspaces (time / freq), C32F<br>built once at creation by bluestein_workspace_kernel", XI, y7 + 40, 320, 56, INPUT)
p.edge(s7a, s7b)
p.edge(ws, s7a)
p.edge(dft, s7a, "read / write", dashed=True, exit=(0.5, 1), entry=(1, 0.5))
p.edge(ree, s7b, "read", dashed=True, exit=(0, 0.9), entry=(1, 0.5))
p.edge(s7b, llr, "write", exit=(1, 0.5), entry=(1, 0.5))

down = p.box("→ UCI segmentation · descrambling / de-rate-matching · LDPC decode · CRC", XS, 1680, SW, 40, OPT)
p.edge(g7, down)
p.edge(llr, down, dashed=True, exit=(1, 0.5), entry=(1, 0.5))

legend(p, 40, 1500 + 180)

# ============================== PAGE 2: channel_est standalone ==============================
p2 = Page("channel_est (standalone)", 1500, 760)
p2.text("channel_est/: standalone 1-D MMSE time/frequency channel estimator (not used by the PUSCH receive pipeline)", 40, 10, 1300, 30, 18, True)
p2.text("Public API cuphyChannelEst1DTimeFrequency() in cuphy.h / cuphy.cpp; used by cuPHY/examples/channel_estimation/cuphy_ex_channel_est.cpp", 40, 42, 1300, 20)

y = 110
iy = p2.tensor("<b>Y</b> received symbols<br>C32F/C16F [Nsc, Nsym, nAnt]", 40, y, 330, 54, INPUT)
iwf = p2.tensor("<b>W_freq</b> frequency filter<br>rank 3 [NscPerUE, NFP=32, nUEs]", 40, y + 75, 330, 54, INPUT)
iwt = p2.tensor("<b>W_time</b> time filter<br>rank 3, per UE", 40, y + 150, 330, 54, INPUT)
iif = p2.tensor("<b>DMRS_index_freq</b> int16 [NFP=32, nUEs]", 40, y + 225, 330, 44, INPUT)
iit = p2.tensor("<b>DMRS_index_time</b> int16 [NTP=4, nUEs]", 40, y + 290, 330, 44, INPUT)

a1 = p2.box("<b>cuphyChannelEst1DTimeFrequency()</b><br>cuphy.cpp: validates args, wraps tensors", 470, y + 20, 360, 56)
a2 = p2.box("<b>channel_est::mmse_1D_time_frequency()</b><br>dispatch on data type (fp32 / fp16)", 470, y + 110, 360, 56)
a3 = p2.box("<b>mmse_1D_time_frequency_launch()</b><br>grid = nUEs blocks · block = 4×4 warps = 512 threads", 470, y + 200, 360, 56)
a4 = p2.box("<b>mmse_1D_time_frequency_kernel</b><br>1 thread block per UE:<br>① load DMRS pilots (freq/time indices)<br>② frequency MMSE interpolation (W_freq)<br>③ time interpolation over 14 symbols (W_time)<br>④ write Hinterp", 470, y + 290, 360, 120, align="left")
p2.edge(a1, a2); p2.edge(a2, a3); p2.edge(a3, a4)
for t in (iy, iwf, iwt, iif, iit):
    p2.edge(t, a1, dashed=True, color="#9673a6", exit=(1, 0.5), entry=(0, 0.5))
out = p2.tensor("<b>Hinterp</b> interpolated channel<br>same type as Y · [NscPerUE, Nsym, nAnt, nUEs]", 930, y + 320, 360, 60)
p2.edge(a4, out, "write")
p2.text("Compare with the PUSCH pipeline (page 1): ch_est/ uses per-UE-group descriptors, DMRS-specific kernels and writes HEst [nRxAnt, nLayers, NF, NH].",
        470, y + 440, 820, 40, 10)
legend(p2, 40, y + 400)

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write('<mxfile host="app.diagrams.net">' + p.xml() + p2.xml() + "</mxfile>")
print("wrote", OUT)
