import os
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuPHY_DL_UL_functional_flow.drawio")

BW, BH, GAP = 150, 64, 26          # step box size / gap
LANE_LABEL_W = 120
LANE_H = 96
LANE_GAP = 18

def q(s):
    return escape(s, {'"': "&quot;"})

class Page:
    def __init__(self, name):
        self.name, self.cells, self.n = name, [], 0

    def nid(self):
        self.n += 1
        return f"{self.name[:2]}{self.n}"

    def box(self, label, x, y, w, h, fill="#dae8fc", stroke="#6c8ebf", extra="", parent="1"):
        i = self.nid()
        style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=11;{extra}"
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def lane(self, label, x, y, w, h, fill):
        i = self.nid()
        style = (f"swimlane;horizontal=0;startSize={LANE_LABEL_W - 90};html=1;fillColor={fill};"
                 f"swimlaneFillColor={fill};strokeColor=#999999;fontStyle=1;fontSize=12;rounded=1;opacity=60;")
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(self, s, t, label="", dashed=False, extra=""):
        i = self.nid()
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;fontSize=10;"
                 + ("dashed=1;" if dashed else "") + extra)
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" edge="1" parent="1" source="{s}" target="{t}">'
                          f'<mxGeometry relative="1" as="geometry"/></mxCell>')
        return i

    def text(self, label, x, y, w, h, size=11, bold=False):
        i = self.nid()
        style = f"text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize={size};" + ("fontStyle=1;" if bold else "")
        self.cells.append(f'<mxCell id="{i}" value="{q(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def chain(self, lane_label, x, y, steps, fill, box_fill="#ffffff", box_stroke="#666666", lane_w=None):
        """Draws a lane with a left-to-right chain of step boxes. Returns (lane_id, [box ids])."""
        w = lane_w or (LANE_LABEL_W - 60 + len(steps) * (BW + GAP) + 10)
        lid = self.lane(lane_label, x, y, w, LANE_H, fill)
        ids = []
        bx = x + LANE_LABEL_W - 60
        for s in steps:
            ids.append(self.box(s, bx, y + (LANE_H - BH) / 2, BW, BH, box_fill, box_stroke))
            bx += BW + GAP
        for a, b in zip(ids, ids[1:]):
            self.edge(a, b)
        return lid, ids

    def xml(self):
        body = "".join(self.cells)
        return (f'<diagram id="{self.name}" name="{self.name}"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" '
                f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2400" '
                f'pageHeight="1400" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}'
                f'</root></mxGraphModel></diagram>')

CP_FILL, CP_STROKE = "#fff2cc", "#d6b656"     # cuPHY-CP / L2
FH_FILL, FH_STROKE = "#f8cecc", "#b85450"     # fronthaul / RU
BUF_FILL, BUF_STROKE = "#e1d5e7", "#9673a6"   # buffers

# =============================== DOWNLINK ===============================
dl = Page("Downlink")
dl.text("cuPHY Downlink functional flow (per slot)", 40, 10, 900, 30, 18, True)
dl.text("L2 → SCF FAPI → cuPHY-CP → cuPHY GPU channel pipelines (batched per cell group) → frequency-domain slot buffer → O-RAN 7.2x fronthaul → RU", 40, 40, 1400, 24)

# left column: L2 + control plane
l2 = dl.box("<b>L2 / MAC</b><br>(scheduler; cuMAC optional)", 40, 100, 170, 70, CP_FILL, CP_STROKE)
fapi = dl.box("<b>SCF FAPI over nvIPC</b><br>DL_TTI.request<br>TX_DATA.request<br>UL_DCI.request", 40, 220, 170, 90, CP_FILL, CP_STROKE)
adp = dl.box("<b>scfl2adapter</b><br>parse FAPI → per-cell slot_command", 40, 360, 170, 70, CP_FILL, CP_STROKE)
drv = dl.box("<b>cuphydriver</b><br>per-slot tick; builds cell-group dyn params; setup() + run()", 40, 480, 170, 90, CP_FILL, CP_STROKE)
dl.edge(l2, fapi); dl.edge(fapi, adp); dl.edge(adp, drv)

LX = 270
lanes_dl = [
    ("PDSCH", "#dae8fc", [
        "TB input<br>(TX_DATA)",
        "TB CRC attach<br>(CRC24A)",
        "CB segmentation<br>+ CB CRC (CRC24B)",
        "LDPC encode<br>BG1/BG2, Zc ≤ 384",
        "Rate matching<br>(rv, LBRM)",
        "Scrambling<br>(Gold seq.)",
        "Modulation<br>QPSK…256QAM",
        "Layer map + precoding<br>(≤ 32 ports)",
        "RE mapping<br>+ DMRS gen, skip CSI-RS REs",
    ]),
    ("PDCCH", "#d5e8d4", [
        "DCI payload<br>(12–140 bits)",
        "CRC24C attach<br>+ RNTI mask",
        "Polar encode<br>(K ≤ 164, N ≤ 512)",
        "Rate matching",
        "Scrambling",
        "QPSK modulation",
        "CCE→REG mapping<br>(interleaved / not)",
        "RE mapping<br>+ PDCCH DMRS",
    ]),
    ("SSB", "#ffe6cc", [
        "MIB (24 bits)",
        "PBCH payload gen<br>(32 bits) + CRC",
        "Polar encode<br>(N = 512)",
        "Rate match (864)<br>+ scrambling",
        "QPSK modulation",
        "PSS / SSS gen<br>(len 127)",
        "PBCH DMRS gen",
        "SSB mapping<br>240 SC × 4 sym",
    ]),
    ("CSI-RS", "#e1d5e7", [
        "RRC params<br>(row 1–18, density)",
        "Sequence gen<br>(Gold, scrambId)",
        "CDM spreading<br>(none/2/4/8)",
        "Optional precoding",
        "RE mapping<br>(≤ 32 ports)",
    ]),
    ("BFW", "#f5f5f5", [
        "SRS channel est.<br>buffers (from UL)",
        "Beamforming weight<br>compute (RZF, λ)",
        "Power normalization",
        "BFP compression<br>(9-bit, optional)",
        "Weights to FH<br>C-plane (ext. 11)",
    ]),
]
max_steps = max(len(s) for _, _, s in lanes_dl)
lane_w = LANE_LABEL_W - 60 + max_steps * (BW + GAP) + 10
y = 100
lane_ids, last_boxes = [], []
for name, fill, steps in lanes_dl:
    lid, ids = dl.chain(name, LX, y, steps, fill, lane_w=lane_w)
    lane_ids.append(lid); last_boxes.append(ids)
    y += LANE_H + LANE_GAP

RX = LX + lane_w + 60
slot = dl.box("<b>DL slot buffer</b><br>per cell, freq-domain IQ<br>(fp16, ports × SC × 14 sym)", RX, 180, 190, 110, BUF_FILL, BUF_STROKE)
fh = dl.box("<b>aerial-fh-driver</b><br>O-RAN 7.2x: C-plane + U-plane<br>BFP compression, packetize", RX, 360, 190, 100, FH_FILL, FH_STROKE)
ru = dl.box("<b>RU</b><br>iFFT + CP insertion,<br>DAC / RF", RX, 530, 190, 90, FH_FILL, FH_STROKE)
dl.edge(slot, fh); dl.edge(fh, ru, "Ethernet / eCPRI")

for lid in lane_ids:
    dl.edge(drv, lid, dashed=True, extra="strokeColor=#b85450;exitX=1;exitY=0.5;")
for ids in last_boxes[:4]:
    dl.edge(ids[-1], slot)
dl.edge(last_boxes[4][-1], fh, "BF weights", extra="strokeColor=#9673a6;")

dl.text("<b>Notes</b><br>• Dashed red arrows: cuphydriver calls setup()/run() on each channel object once per slot; each call batches all cells of the cell group.<br>"
        "• cuPHY works in the frequency domain; iFFT/CP are done in the RU (O-RAN split 7.2x).<br>"
        "• PDSCH rate-matches around co-scheduled CSI-RS REs; CSI-RS itself is also generated by its own pipeline.<br>"
        "• BFW consumes SRS channel estimates produced by the UL SRS pipeline.",
        LX, y + 10, 1100, 90)

# =============================== UPLINK ===============================
ul = Page("Uplink")
ul.text("cuPHY Uplink functional flow (per slot)", 40, 10, 900, 30, 18, True)
ul.text("RU → O-RAN 7.2x fronthaul → per-cell frequency-domain slot buffers → cuPHY GPU channel pipelines (batched per cell group) → cuPHY-CP → SCF FAPI indications → L2", 40, 40, 1500, 24)

ru_u = ul.box("<b>RU</b><br>RF / ADC,<br>CP removal + FFT", 40, 100, 170, 80, FH_FILL, FH_STROKE)
fh_u = ul.box("<b>aerial-fh-driver</b><br>receive U-plane,<br>decompress, order packets", 40, 240, 170, 90, FH_FILL, FH_STROKE)
buf_u = ul.box("<b>UL slot buffers</b><br>per cell, freq-domain IQ<br>(Rx ant × SC × 14 sym)", 40, 390, 170, 100, BUF_FILL, BUF_STROKE)
prach_buf = ul.box("<b>PRACH occasion buffers</b>", 40, 560, 170, 60, BUF_FILL, BUF_STROKE)
ul.edge(ru_u, fh_u, "Ethernet / eCPRI"); ul.edge(fh_u, buf_u); ul.edge(fh_u, prach_buf, extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")

LX = 270
lanes_ul = [
    ("PUSCH<br>data (SCH)", "#dae8fc", [
        "DMRS channel est.<br>(legacy MMSE / multistage<br>/ RKHS / LS)",
        "Noise + interference<br>estimation",
        "CFO / TA estimation<br>(+ correction)",
        "Equalizer coeffs<br>(RZF / MMSE / MMSE-IRC<br>/ shrinkage)",
        "Equalization<br>(+ IDFT for DFT-s-OFDM)",
        "Soft demapping<br>→ LLRs",
        "Descramble, de-rate-match<br>+ HARQ combining",
        "LDPC decode<br>(TB or CB mode)",
        "CRC check<br>→ TB out",
    ]),
    ("PUSCH<br>UCI", "#dae8fc", [
        "UCI LLR segmentation<br>(HARQ / CSI-1 / CSI-2)",
        "Simplex (≤ 2 b) / RM (≤ 11 b)<br>/ Polar list decode",
        "DTX detection",
        "CSI-2 size calc<br>(from CSI-1)",
        "CSI-2 segment<br>+ decode",
    ]),
    ("PUSCH<br>meas.", "#dae8fc", [
        "RSSI",
        "RSRP",
        "SINR / noise var.",
        "TA / CFO report",
    ]),
    ("PUCCH<br>F0 / F1", "#d5e8d4", [
        "F0: sequence<br>correlation",
        "F1: ch. est. +<br>OCC despread",
        "HARQ / SR<br>detection + DTX",
    ]),
    ("PUCCH<br>F2 / F3", "#d5e8d4", [
        "F2 / F3 front end<br>(ch. est., equalize;<br>F3: IDFT)",
        "LLR + descramble",
        "UCI segmentation<br>(F234 UCI seg)",
        "Simplex / RM /<br>Polar decode",
        "CSI-P2 control<br>+ decode (F3)",
    ]),
    ("PRACH", "#ffe6cc", [
        "Occasion IQ<br>(format 0 / B4)",
        "FFT correlation with<br>ZC root seq. (256/1024)",
        "IFFT → power<br>delay profile",
        "Non-coherent combining<br>(antennas, repetitions)",
        "Peak search<br>vs. threshold",
        "Preamble idx, delay (TA),<br>power, RSSI, interference",
    ]),
    ("SRS", "#e1d5e7", [
        "SRS symbols<br>(comb 2/4, ≤ 4 sym)",
        "Channel est.<br>(MMSE / RKHS)",
        "Delay offset corr.<br>+ normalization",
        "ChEst buffers<br>(per UE, per PRG)",
        "SRS report<br>(→ L2) & → DL BFW",
    ]),
]
max_steps = max(len(s) for _, _, s in lanes_ul)
lane_w = LANE_LABEL_W - 60 + max_steps * (BW + GAP) + 10
y = 100
B, ul_lanes = [], []
for name, fill, steps in lanes_ul:
    lid, ids = ul.chain(name, LX, y, steps, fill, lane_w=lane_w)
    ul_lanes.append(lid); B.append(ids)
    y += LANE_H + LANE_GAP
PUSCH, UCI, MEAS, F01, F23, PRACH, SRS = range(7)

# inputs from slot buffers
for k in (PUSCH, F01, F23, SRS):
    ul.edge(buf_u, B[k][0], dashed=True, extra="strokeColor=#9673a6;")
ul.edge(prach_buf, B[PRACH][0], dashed=True, extra="strokeColor=#9673a6;")

# cross-lane dependencies inside PUSCH
ul.edge(B[PUSCH][5], B[UCI][0], "UCI LLRs", extra="strokeColor=#6c8ebf;")
ul.edge(B[PUSCH][0], B[MEAS][0], "DMRS / ch. est.", dashed=True, extra="strokeColor=#6c8ebf;")

# right column: control plane back to L2
RX = LX + lane_w + 60
drv_u = ul.box("<b>cuphydriver</b><br>collects per-cell results<br>after run()", RX, 180, 190, 90, CP_FILL, CP_STROKE)
adp_u = ul.box("<b>scfl2adapter</b><br>build FAPI indications", RX, 340, 190, 70, CP_FILL, CP_STROKE)
fapi_u = ul.box("<b>SCF FAPI over nvIPC</b><br>CRC.indication<br>RX_DATA.indication<br>UCI.indication<br>RACH.indication<br>SRS.indication", RX, 470, 190, 130, CP_FILL, CP_STROKE)
l2_u = ul.box("<b>L2 / MAC</b>", RX, 660, 190, 60, CP_FILL, CP_STROKE)
ul.edge(drv_u, adp_u); ul.edge(adp_u, fapi_u); ul.edge(fapi_u, l2_u)
for k in range(7):
    ul.edge(B[k][-1], drv_u, dashed=True, extra="strokeColor=#d6b656;")

ul.text("<b>Notes</b><br>• cuphydriver issues setup()/run() per channel object each slot, batching all cells of the cell group (UL_TTI.request drives what is scheduled).<br>"
        "• FFT/CP removal are done in the RU (O-RAN split 7.2x); cuPHY starts from frequency-domain IQ.<br>"
        "• Early-HARQ: PUSCH can run a sub-slot path on the first DMRS/data symbols to decode HARQ-ACK before the full slot arrives.<br>"
        "• SRS channel estimates are kept in GPU buffers and reused by the DL BFW pipeline.",
        LX, y + 10, 1300, 90)

with open(OUT, "w", encoding="utf-8") as fh_out:
    fh_out.write('<mxfile host="app.diagrams.net">' + dl.xml() + ul.xml() + "</mxfile>")
print("wrote", OUT)
