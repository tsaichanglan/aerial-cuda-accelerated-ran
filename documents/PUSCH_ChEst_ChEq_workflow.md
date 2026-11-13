# PUSCH Channel Estimation and Equalization: Kernel Workflow

This document traces, kernel by kernel, how cuPHY turns received PUSCH IQ samples into LLRs:

- **`ch_est/`**: the PUSCH channel estimator used by the receive pipeline.
- **`channel_eq/`**: equalizer coefficients, equalization, soft demapping, and the DFT-s-OFDM IDFT.
- **`channel_est/`**: a separate, standalone 1-D MMSE estimator. It is **not** used by the PUSCH receive pipeline (see [Part 2](#part-2--channel_est-standalone-1-d-mmse-estimator)).

Two stages that sit between estimation and equalization (noise/interference estimation and CFO/TA estimation) are included because they produce inputs the equalizer needs.

The diagram is in [PUSCH_ChEst_ChEq_workflow.drawio](PUSCH_ChEst_ChEq_workflow.drawio). Source paths are relative to this folder.

## Notation

Tensor shapes are written `[d0, d1, …]` with **d0 the fastest-changing (contiguous) dimension**, as cuPHY tensors are column-major. Shapes come from the allocations in `PuschRx` ([pusch_rx.cpp](../cuPHY/src/cuphy_channels/pusch_rx.cpp), per-UE-group tensor setup) and from `cuphyPuschRxUeGrpPrms_t` in [cuphy.h](../cuPHY/src/cuphy/cuphy.h).

All tensors except `DataRx` are **per UE group**. A UE group is the set of UEs sharing the same PRBs and symbols (MU-MIMO).

| Symbol | Meaning |
|---|---|
| `nRxAnt` | gNB receive antennas for the cell |
| `nLayers` | total layers in the UE group (≤ 8 per UE group) |
| `nPrb` | PRBs allocated to the UE group |
| `NF` | subcarriers of the UE group = 12 × `nPrb` |
| `NF_cell` | subcarriers of the cell's UL bandwidth part = 12 × `nPrbUlBwp` |
| `NH` | number of channel estimates in time = `dmrsAddlnPos` + 1 (1–4) |
| `nDmrsSyms` | DMRS symbols in the slot (1–4) |
| `Nd` | data symbols; includes the DMRS symbols when `nDmrsCdmGrpsNoData` = 1 (data shares DMRS symbols) |
| `nUes`, `nUeGrps` | UEs / UE groups in the cell group |
| `C16F`, `C32F`, `R16F`, `R32F` | complex/real, 16-bit/32-bit float |

## Execution order per slot

The full-slot path in `PuschRx::fullSlotKernelLaunch()` ([pusch_rx.cpp](../cuPHY/src/cuphy_channels/pusch_rx.cpp)) launches these stages in order on one stream. Each stage may launch several kernels when UE groups need different configurations ("heterogeneous configs").

| Step | Stage | Directory | Runs when |
|---|---|---|---|
| 1 | Channel estimation | `ch_est/` | always; once per time instance (up to `NH`) |
| 2 | Noise and interference estimation | `pusch_noise_intf_est/` | SINR measurement enabled, or an MMSE-type equalizer is selected |
| 3 | RSRP measurement | `pusch_rssi/` | SINR measurement and weighted-average CFO both enabled |
| 4 | CFO / timing-offset estimation | `cfo_ta_est/` | always (correction applied only if enabled) |
| 5 | Equalizer coefficient computation | `channel_eq/` | always; once per time instance (up to `NH`) |
| 6 | Equalization + soft demapping | `channel_eq/` | always |
| 7 | DFT-s-OFDM: IDFT, then soft demapping | `channel_eq/` | DFT-s-OFDM enabled |

After step 6/7 the LLRs go to UCI segmentation, de-rate-matching and LDPC decoding (not covered here).

Early-HARQ (sub-slot) processing runs the same stages on the first DMRS symbol(s) only (`PuschRx::subSlotKernelLaunch()`), with the soft demapper limited to symbols 0–3 (bitmask `0xF`, or `0x1F` with an extra DMRS symbol).

---

## Part 1 — PUSCH receive pipeline

### Input tensors

| Tensor | Type | Shape | Notes |
|---|---|---|---|
| `DataRx` (`tInfoDataRx`) | C16F (C32F if built with `ASIM_CUPHY_FP32`) | `[NF_cell, 14, nRxAnt]` | Per **cell** slot buffer from the fronthaul; each UE group reads its PRBs and symbols from it. |
| Frequency-interpolation filters (`pWFreq`, `pWFreq4`, `pWFreqSmall`) | real | rank 3: `[output tone (incl. grid shift), input DMRS tone, filter index]` | Static MMSE interpolation filters, given at PUSCH object creation. Variants for the 8-, 4- and 1–3-PRB windows. Filter index 0 = middle of the band, 1 = lower edge, 2 = upper edge. |
| Shift sequence (`pShiftSeq`, `pShiftSeq4`) | complex | `[nPrb·6, nDmrsSyms]` | Static phase ramp that centers the channel delay before filtering (legacy algorithm). |
| Unshift sequence (`pUnShiftSeq`, `pUnShiftSeq4`) | complex | `[nPrb·12·2 + 1]` | Opposite ramp applied after filtering; the `+1` is the frequency offset between the two DMRS grids. |
| RKHS parameters (`pPuschRkhsPrms`) | — | — | Static; only for the RKHS estimator. |
| UE-group parameters (`cuphyPuschRxUeGrpPrms_t`) | struct | one per UE group | DMRS positions, ports, scrambling ID, `nPrb`, `startPrb`, QAM order per layer, algorithm flags. |

### Step 1 — Channel estimation (`ch_est/`)

Selected by `cuphyPuschStatPrms_t.chEstAlgo`. Kernel selection is in `puschRxChEstKernelBuilder::kernelSelectL0/L1` in [ch_est.cu](../cuPHY/src/cuphy/ch_est/ch_est.cu); launching is in `ChestStream::launchKernels()` in [ch_est_stream.cpp](../cuPHY/src/cuphy/ch_est/ch_est_stream.cpp).

The kernel variant also depends on the UE group's PRB count:

| PRB count | Thread-block tiling | Kernel family |
|---|---|---|
| `nPrb` ≥ 8 and a multiple of 4 | 8 DMRS PRBs in → 4 PRBs of estimates out | windowed |
| `nPrb` ≥ 4 otherwise (incl. exactly 4) | 4 DMRS PRBs in → 2 PRBs out | windowed |
| `nPrb` = 1, 2 or 3 | whole allocation | small |

**How many thread blocks run.** The launch grid is sized by the PRBs each block *writes*, not the PRBs it reads (`puschRxChEstKernelBuilder::computeKernelLaunchGeo()` in [ch_est.cu](../cuPHY/src/cuphy/ch_est/ch_est.cu)):

```cpp
N_THREAD_BLKS_PER_BS_ANT = div_round_up(nPrb, N_DMRS_INTERP_PRB_OUT_PER_CLUSTER);  // PRBs written per block
gridDim  = dim3(N_THREAD_BLKS_PER_BS_ANT, nRxAnt, nUeGrps);
blockDim = dim3(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER × 2 grids × 12 tones);           // = 96 for 8-in / 4-out
```

Each block therefore reads a window of 8 PRBs of DMRS but writes only 4 PRBs of `HEst`, so the input windows overlap. Middle blocks read 2 extra PRBs on each side as context for the interpolation filter. The first and last blocks cannot extend past the allocation, so their windows are pinned to the band edges and they use the lower- and upper-edge filters.

*Example: one UE group with `nPrb` = 32, legacy MMSE.* 32 is ≥ 8 and a multiple of 4, so the 8-in / 4-out kernel is used, with 32 / 4 = **8 blocks per antenna** (not 32 / 8 = 4):

| Block (`blockIdx.x`) | Reads DMRS PRBs | Writes `HEst` PRBs | Filter |
|---|---|---|---|
| 0 | 0–7 | 0–3 | lower edge |
| 1 | 2–9 | 4–7 | middle |
| 2 | 6–13 | 8–11 | middle |
| *i* (middle) | 4*i*−2 … 4*i*+5 | 4*i* … 4*i*+3 | middle |
| 6 | 22–29 | 24–27 | middle |
| 7 | 24–31 | 28–31 | upper edge |

- **Blocks per launch:** 8 × `nRxAnt`. With 4 receive antennas the grid is `(8, 4, 1)`: 32 blocks of 96 threads, all in **one** kernel launch.
- **Launches per slot:** one per DMRS time position, i.e. `NH` = `dmrsAddlnPos` + 1. With one additional DMRS position, the kernel is launched twice.
- **Multi-stage algorithm:** kernel ① uses 4-PRB clusters (`CUPHY_PUSCH_RX_CH_EST_NON_PRG_DELAY_EST_PRB_CLUSTER_SIZE`), so its grid x is also `div_round_up(32, 4)` = 8. With per-PRG estimation enabled it uses 12-PRB clusters instead, giving `div_round_up(32, 12)` = 3.

*Example: one UE group with `nPrb` = 34, legacy MMSE.* 34 is not a multiple of 4, so the narrower 4-in / 2-out kernel is used: each block reads a 4-PRB window (1 extra PRB of context on each side) and writes 2 PRBs. That gives `div_round_up(34, 2)` = **17 blocks per antenna** of 2 PRBs × 2 grids × 12 tones = **48 threads**, grid `(17, nRxAnt, nUeGrps)`, still one launch per DMRS time position.

Block *i* always writes PRBs 2*i* and 2*i*+1. Its input window starts at 2*i*−1, except at the two ends ([ch_est.cu](../cuPHY/src/cuphy/ch_est/ch_est.cu), `prbClusterStartIdx` and `INTERP_DMRS_ABS_TONE_IDX` in `windowedChEstNoDftSOfdmKernel`):

| Block (`blockIdx.x`) | Reads DMRS PRBs | Writes `HEst` PRBs | Filter |
|---|---|---|---|
| 0 | 0–3 | 0–1 | lower edge |
| 1 | 1–4 | 2–3 | middle |
| 2 | 3–6 | 4–5 | middle |
| *i* (middle) | 2*i*−1 … 2*i*+2 | 2*i* … 2*i*+1 | middle |
| 15 | 29–32 | 30–31 | middle |
| 16 | 30–33 | 32–33 | upper edge |

17 blocks × 2 PRBs = 34, so the allocation is covered exactly.

| | `nPrb` = 32 | `nPrb` = 34 |
|---|---|---|
| Window (read → write) | 8 → 4 PRBs | 4 → 2 PRBs |
| Blocks per antenna | 8 | 17 |
| Threads per block | 96 | 48 |
| Extra PRBs on each side for the filter | 2 | 1 |

Adding 2 PRBs switches the kernel to the narrower window: roughly twice as many blocks, and each output PRB is estimated with less frequency context.

**Odd PRB counts** (only in the 4-in / 2-out kernel), e.g. `nPrb` = 35:

- **Grid:** `div_round_up(35, 2)` = 18 blocks per antenna, which would naively cover 36 PRBs.
- **Last block:** reads PRBs 31–34 and shifts its output down by one PRB, so it covers PRBs 33–34.
- **No double write:** it then drops its first output PRB and writes only PRB 34, because PRB 33 was already written by block 16.

Both adjustments are keyed on `nPrbsMod2 == 1` with the upper-edge filter in the kernel.

**Multi-stage algorithm:** kernel ① has a fixed 4-PRB cluster, so for `nPrb` = 34 its grid x is `div_round_up(34, 4)` = 9, or `div_round_up(34, 12)` = 3 with per-PRG estimation. Kernel ② sizes grid x from the largest PRB-cluster count across the UE groups and may group several antennas per block in y. Each kernel is launched once per DMRS time position.

*Example: two UEs, four layers, single-symbol DMRS.* One UE group with `nPrb` = 32, DMRS type 1, `nDmrsCdmGrpsNoData` = 2, one DMRS symbol, no additional DMRS. UE1 uses ports 0 and 1, UE2 uses ports 2 and 3. Legacy MMSE. The example assumes `nRxAnt` = 4; the selector requires `nRxAnt` ≥ `nLayers`.

**Kernel selection.** Both UEs are in the same UE group, so `nLayers` = 4: UE1 has layers 0–1 and UE2 has layers 2–3.

- `kernelSelectL1` takes the single-symbol branch: `nLayers` ≤ 4, 2 grids, `nDmrsSyms` == 1, `Nh` == 1 ([ch_est.cu](../cuPHY/src/cuphy/ch_est/ch_est.cu)).
- That branch instantiates `N_LAYERS` = 4, `N_DMRS_GRIDS_PER_PRB` = 2, `N_DMRS_SYMS` = 1.
- It supports at most 4 layers: a single DMRS symbol has no time OCC, so 2 grids × 2 fOCC = 4.
- `nPrb` = 32 selects the 8-in / 4-out `windowedChEstNoDftSOfdmKernel`.

**Port mapping on the DMRS symbol.**

| Port | UE | Grid (CDM group) | Subcarriers | fOCC |
|---|---|---|---|---|
| 0 | UE1 | 0 | even | +1 +1 |
| 1 | UE1 | 0 | even | +1 −1 |
| 2 | UE2 | 1 | odd | +1 +1 |
| 3 | UE2 | 1 | odd | +1 −1 |

With `nDmrsCdmGrpsNoData` = 2, both combs carry DMRS and the DMRS symbol carries no data, so the √2 scaling is not applied.

**Input.**

| Tensor | Shape | What the kernel reads |
|---|---|---|
| `DataRx` | `[NF_cell, 14, nRxAnt]` (whole cell slot) | 1 DMRS symbol × 32 PRBs × 12 subcarriers = 384 REs per antenna (192 per grid); 1,536 REs for 4 antennas |
| Frequency filters `pWFreq` | `[output tone, input tone, 3]` | the 8-PRB filters: middle, lower edge, upper edge |
| Shift sequence | `[32·6, 1]` = `[192, 1]` | |
| Unshift sequence | `[32·12·2 + 1]` = `[769]` | |

**Output.**

| Tensor | Shape | Size |
|---|---|---|
| `HEst` | `[nRxAnt, nLayers, NF, NH]` = `[4, 4, 384, 1]` | 4 × 4 × 384 = 6,144 complex values |

- NF = 32 × 12 = 384. NH = 1 because there is no additional DMRS.
- Every layer is estimated on all 12 subcarriers of each PRB, including the subcarriers of the other comb. Each grid's 6 pilots per PRB are interpolated to 12 tones.

**Launch.**

| | Value | Why |
|---|---|---|
| Launches | 1 | NH = 1 |
| `gridDim` | `(8, 4, 1)` | div_round_up(32, 4) = 8 blocks per antenna, × `nRxAnt`, × 1 UE group |
| Blocks | 32 | 8 × 4 |
| `blockDim` | 96 | 4 output PRBs × 2 grids × 12 tones |
| Threads | 3,072 | 32 × 96 |

**One block** (one antenna and one 8-PRB window; e.g. block 1 reads PRBs 2–9 and writes PRBs 4–7):

1. Each of the 96 threads loads one subcarrier of the DMRS symbol: 48 pilots for grid 0 and 48 for grid 1.
2. Each grid is descrambled and its delay centered. Then the frequency filter separates that grid's 2 fOCC layers and interpolates them: grid 0 gives ports 0 and 1, grid 1 gives ports 2 and 3.
3. Each output thread owns one interpolated tone of one grid and writes the layers that belong to its grid, 2 layers per thread. The output-write loop checks `(OCCIdx[i] >> 2) & 1 == DMRS_GRID_IDX`. The block writes 4 PRBs × 12 tones × 4 layers = 192 `HEst` values.

The two UEs need no separate work: their ports are in different CDM groups of the same UE group, so one launch covers both. A second UE group would add a z-slice to the grid (`gridDim.z` = 2), not another launch.

*Same example with one additional DMRS position (`dmrsAddlnPos` = 1, so NH = 2).* The kernel is launched **twice**, once per DMRS time position. The two launches are identical apart from the DMRS symbol they read and the `HEst` slice they write.

**Why two launches.**

- Setup loops over `chEstTimeInstIdx` = 0 … `CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST`−1 and builds a separate set of launch configurations for each time instance ([ch_est.cu](../cuPHY/src/cuphy/ch_est/ch_est.cu), `setup`).
- A UE group joins instance *t* only if `t ≤ dmrsAddlnPos` (`if(chEstTimeInstIdx > dmrsAddlnPos) continue;` in `batch`). With `dmrsAddlnPos` = 1 it joins instances 0 and 1.
- The kernel selector is always called with `Nh = 1` hard-coded, so the `1 == Nh` condition in `kernelSelectL1` is always met. NH is handled by launching once per instance, not inside the kernel.
- Each launch's descriptor carries `chEstTimeInst = t`. The kernel reads that position's DMRS symbol from `dmrsSymLoc[t·dmrsMaxLen]` and writes only the slice `HEst(:, :, :, NH_IDX = t)`.

**Launch summary** (single-symbol DMRS, typically on symbols 2 and 11; `nRxAnt` = 4):

| | Launch 0 | Launch 1 | Total |
|---|---|---|---|
| DMRS symbol read | first position (e.g. symbol 2) | additional position (e.g. symbol 11) | |
| `gridDim` | `(8, 4, 1)` | `(8, 4, 1)` | |
| `blockDim` | 96 | 96 | |
| Blocks / threads | 32 / 3,072 | 32 / 3,072 | 64 / 6,144 |
| Writes | `HEst[:, :, :, 0]` | `HEst[:, :, :, 1]` | `HEst` = `[4, 4, 384, 2]` = 12,288 values |

**How the launches are issued.**

- **Stream mode:** `launchKernelsImpl` enqueues the instances one after another on the same stream ([ch_est_stream.cpp](../cuPHY/src/cuphy/ch_est/ch_est_stream.cpp)), so they run in sequence.
- **Graph mode:** there is one kernel node per instance ([ch_est_graph_mgr.cpp](../cuPHY/src/cuphy/ch_est/ch_est_graph_mgr.cpp), `addChestKernelImpl`). All of them are added from the same predecessor list, so they appear able to run concurrently (the dependency wiring inside `addKernelImpl` has not been checked). Nodes for unused instances (here 2 and 3) are disabled for the slot.
- **Multi-stage algorithm:** kernels ① and ② are each issued once per instance, so NH = 2 means 2 + 2 launches.
- **Early-HARQ:** instance 0 has already run in the sub-slot pass, so the full-slot pass starts at instance 1 (`startChEstInstIdx`).

Kernels with `NoDftSOfdm` in the name are used when DFT-s-OFDM is disabled; the versions without it handle low-PAPR DMRS for DFT-s-OFDM.

| Algorithm (`chEstAlgo`) | Kernel(s), in order | Inputs | Outputs |
|---|---|---|---|
| **0 — Legacy MMSE** | `windowedChEstNoDftSOfdmKernel` / `windowedChEstKernel` (≥ 4 PRBs), or `smallChEstNoDftSOfdmKernel` / `smallChEstKernel` (1–3 PRBs) | `DataRx`, frequency filters, shift sequences | `HEst` |
| **1 — Multi-stage MMSE with delay estimation** | ① `windowedChEstPreNoDftSOfdmKernel`: least-squares estimate on DMRS tones, accumulates delay statistics<br>② `chEstFilterNoDftSOfdmDispatchKernel` (or `chEstFilterPrgNoDftSOfdmDispatchKernel` when per-PRG estimation is enabled): delay compensation and MMSE filtering | ① `DataRx`<br>② `DmrsLSEst`, `DmrsAccum` | ① `DmrsLSEst`, `DmrsAccum`<br>② `DmrsDelayMean`, `HEst` |
| **2 — RKHS** | `puschRkhsChEstKernel<nZpDmrsSc, log2numZpDmrsSc>`; zero-padded size chosen from the 6 × `nPrb` DMRS subcarriers | `DataRx`, RKHS parameters | `HEst` |
| **3 — LS only** | `windowedChEstPreNoDftSOfdmKernel` only (no filtering kernel) | `DataRx` | `DmrsLSEst` (this kernel does not write `HEst`; the LS estimates feed the TensorRT path below) |
| **TensorRT model** (via the channel-estimator factory, `chest_factory.*`, `trtengine_chest.cpp`) | ① LS kernel as above<br>② `prepareChestMlInputsKernel`<br>③ TensorRT engine<br>④ `extractChestMlOutputsKernel` (② and ④ in `ch_est_trtengine_pre_post_converion.cu`) | ② `DmrsLSEst` | ④ `HEst` |

RKHS is used only when `HEst` is C32F, `DataRx` is C16F and `nRxAnt` ≤ 4.

With early-HARQ and the multi-stage algorithm, kernel ① for the first DMRS symbol already ran in the sub-slot path, so the full-slot path starts from the second time instance; kernel ② still runs for all instances.

**Channel-estimation tensors**

| Tensor | Type | Shape | Produced by | Consumed by |
|---|---|---|---|---|
| `HEst` (`tInfoHEst`) | C32F | `[nRxAnt, nLayers, NF, NH]` | step 1 | steps 2, 3, 4, 5 |
| `DmrsLSEst` (`tInfoDmrsLSEst`) | C32F | `[NF/2, nLayers, nRxAnt, NH]` | step 1 ① | step 1 ② |
| `DmrsAccum` (`tInfoDmrsAccum`) | C32F | `[2]` (double-buffered: active and inactive) | step 1 ① | step 1 ② |
| `DmrsDelayMean` (`tInfoDmrsDelayMean`) | R32F | `[NH]` | step 1 ② | — (delay shift; also kept for debugging) |
| `ChEstDbg` (debug only) | C32F | `[NF/2, nDmrsSyms, 1, 1]` | step 1 | debug output |

`NF/2` is the number of DMRS subcarriers per layer for DMRS configuration type 1 (every other subcarrier).

#### What channel estimation calculates

ch_est estimates the uplink channel matrix **H** for every receive antenna, layer and subcarrier of a UE group, from the DMRS pilots.

**Signal model.** On DMRS subcarrier *k*, DMRS symbol *l* and receive antenna *r*:

```
Y[k, l, r] = Σ_layer  H[r, layer, k] · x_layer[k, l]  +  noise
```

The pilot `x_layer` is known: a QPSK Gold-sequence symbol (TS 38.211 §6.4.1.1, seeded from `dmrsScrmId`, `scid` and the slot/symbol number) multiplied by orthogonal cover codes. Up to 8 layers share the same resource elements and are separated by:

| Separation | How | Layers |
|---|---|---|
| DMRS grid (CDM group) | even vs. odd subcarriers | × 2 |
| Frequency OCC | `[+1, +1]` / `[+1, −1]` over adjacent pilot subcarriers of a grid | × 2 |
| Time OCC | `[+1, +1]` / `[+1, −1]` over the two symbols of a double-symbol DMRS (`dmrsMaxLen` = 2) | × 2 |

Each layer's grid and cover codes come from its DMRS port, via `OCCIdx` in the UE-group parameters.

**Processing steps** (legacy MMSE; the other algorithms reuse steps 1–3):

1. **Descramble.** Multiply each pilot by the conjugate of its Gold-sequence QPSK value `(±1 ± j)/√2`, leaving the channel times the cover codes.
2. **Remove the time OCC.** Average the two DMRS symbols with signs `(+,+)` and `(+,−)`; this separates layers by time code.
3. **Remove the frequency OCC.** Combine adjacent pilot subcarriers with `(+,+)` and `(+,−)`; this separates layers by frequency code. The result is one least-squares (LS) estimate per layer on every other subcarrier (`DmrsLSEst`).
4. **Delay centering.** Multiply by a phase ramp `e^{+j2π·f·τ}` so the channel's delay sits where the interpolation filter expects it.
5. **Frequency interpolation and denoising.** Matrix-vector product `H = W · Y_LS`. `W` is an MMSE filter mapping the pilot estimates of a window of PRBs onto all 12 subcarriers of each PRB; separate filters handle the band edges.
6. **Undo the delay centering** with the opposite ramp `e^{−j2π·f·τ}`, then write `HEst`. When `nDmrsCdmGrpsNoData` = 1 the estimate is also scaled by √2.

In time, ch_est produces one estimate per DMRS position (`NH` of them, one kernel launch each). Interpolation between DMRS positions for the data symbols happens later, in the equalizer (`enablePuschTdi`).

**Per algorithm**

- **0 · Legacy MMSE.** Steps 1–6 in one kernel, using the static shift sequence (fixed τ).
  - Grid `(PRB clusters, nRxAnt, UE groups)`, one thread per DMRS subcarrier. An 8-PRB window has 6 pilot tones × 8 PRBs × 2 grids = 96 threads.
    - *Grid* here means a DMRS CDM group, one comb of DMRS configuration type 1. Grid 0 is the even subcarriers (CDM group 0: ports 0/1, plus 4/5 with tOCC). Grid 1 is the odd subcarriers (CDM group 1: ports 2/3, plus 6/7). Each grid has 6 pilots per PRB, so 6 × 2 = 12: one thread for every subcarrier of the window. Layers = grids × fOCC × tOCC = 2 × 2 × 2 = 8 at most. Each grid's 6 pilots per PRB are interpolated to all 12 subcarriers, which is why the output side is also 4 PRBs × 12 tones × 2 grids = 96 ([ch_est.cu](../cuPHY/src/cuphy/ch_est/ch_est.cu), `N_DMRS_GRIDS_PER_PRB`, `N_DMRS_TONES`).
  - **Overlapping windows:** each thread block reads 8 PRBs of pilots but writes only the middle 4 PRBs of estimates (4-PRB windows: reads 4, writes 2). That is why separate edge filters exist.
- **1 · Multi-stage MMSE with delay estimation.** Same steps, but τ is **measured** instead of fixed.
  - Kernel ① does steps 1–3, writes `DmrsLSEst`, and accumulates the correlation between neighbouring pilots, `R1 = Σ conj(H_k) · H_{k+1}`, into `DmrsAccum` (summed across thread blocks with atomic adds into the active buffer, while the inactive buffer is cleared for its next use).
  - Kernel ② computes the mean delay
    ```
    τ = −arg(R1) / (2π · P · Δf),   Δf = 15 kHz · 2^μ
    ```
    where `P` = 2 for one layer, or 4 when adjacent tones were first averaged. It writes τ to `DmrsDelayMean`, builds the shift and unshift ramps from τ, and applies steps 4–6 to `DmrsLSEst`. The per-PRG variant computes this per precoding resource group.
- **2 · RKHS.** A reproducing-kernel-Hilbert-space estimator (`puschRkhsChEstKernel`):
  - Projects the descrambled pilots onto precomputed eigenvectors of the channel correlation (`eigenVecs`, `eigenValues`, `eigenCorr` in `cuphyRkhsGridPrms_t`), using a Hamming window and FFTs.
  - Measures the noise energy.
  - Runs **matching pursuit**: repeatedly selects the strongest delay component and updates the coefficients until the remaining energy reaches the noise level.
  - Reconstructs the channel on all subcarriers.
- **3 · LS only.** Steps 1–3 only; writes `DmrsLSEst`.
- **TensorRT.** Reuses the LS output: `prepareChestMlInputsKernel` reformats `DmrsLSEst` for the network, the TensorRT engine runs the model, and `extractChestMlOutputsKernel` writes the result into `HEst`.

**Setup and run**

1. **Setup (CPU, per slot).** The kernel builder (`kernelSelectL0/L1` in `ch_est.cu`) picks one kernel per group of UE groups with the same configuration. The choice depends on `chEstAlgo`, `nPrb`, DMRS symbol count, layer count and DFT-s-OFDM. Kernel parameters are written into GPU descriptors.
2. **Run.** `ChestStream::launchKernels()` launches once per DMRS time position:
   - **Legacy, RKHS, LS only:** one kernel per position.
   - **Multi-stage:** kernel ① for all positions, then kernel ② for all positions, so τ uses every DMRS symbol.
3. **Consumers.** `HEst` is read by noise/interference estimation, CFO/timing estimation, RSRP measurement and equalizer-coefficient computation (steps 2–5 below).

### Step 2 — Noise and interference estimation (`pusch_noise_intf_est/`)

| Kernel | Inputs | Outputs |
|---|---|---|
| `noiseIntfEstNoDftSOfdmKernel` / `noiseIntfEstKernel` | `DataRx` (DMRS symbols), `HEst` | `NoiseVarPreEq`, `LwInv` |

| Tensor | Type | Shape | Notes |
|---|---|---|---|
| `NoiseVarPreEq` (`tInfoNoiseVarPreEq`) | R32F | `[nUes]` (`[nUeGrps]` in some configurations) | pre-equalizer noise-plus-interference power |
| `LwInv` (`tInfoLwInv`) | C32F | `[nRxAnt, nRxAnt, nPrb]` | inverse Cholesky factor of the per-PRB noise-interference covariance; allocated only for MMSE-type equalizers or SINR measurement |

### Step 3 — RSRP (optional, `pusch_rssi/`)

Uses `HEst`; writes `Rsrp` (R32F, `[nUes]`). RSSI (`[nUeGrps]`, plus a per-symbol/per-antenna `RssiFull` of `[14, 64, nUeGrps]`) is measured from `DataRx` in the same directory.

### Step 4 — CFO and timing-offset estimation (`cfo_ta_est/`)

| Kernel | Inputs | Outputs |
|---|---|---|
| `cfoTaEstLowMimoKernel` | `HEst` (estimates at different DMRS times) | `CfoEst`, `CfoPhaseRot`, `TaPhaseRot`, `CfoHz`, `TaEst` |

| Tensor | Type | Shape | Notes |
|---|---|---|---|
| `CfoEst` (`tInfoCfoEst`) | C32F | `[14, nUes]` | per-symbol CFO phase per UE |
| `CfoPhaseRot` | C32F | `[4, 8, maxUeGrps]` (`[NH max, layers max, UE groups max]`) | time-domain phase rotation applied by the equalizer |
| `TaPhaseRot` | C32F | `[8, nUeGrps]` | frequency-domain phase rotation for timing offset |
| `CfoHz`, `TaEst` | R32F | `[nUes]` | reported CFO (Hz) and timing offset |

### Step 5 — Equalizer coefficients (`channel_eq/`)

Runs once per time instance (up to `NH`; up to 4 with time-domain interpolation, `enablePuschTdi`). The coefficient math is selected by `eqCoeffAlgo`:

| `eqCoeffAlgo` | Method | Uses `LwInv` |
|---|---|---|
| 0 | RZF (regularized zero-forcing) | no |
| 1 | MMSE with diagonal noise | no |
| 2 | MMSE-IRC (interference rejection combining) | yes |
| 3 | MMSE-IRC with RBLW covariance shrinkage | yes |
| 4 | MMSE-IRC with OAS covariance shrinkage | yes |

The kernel is selected by antenna and layer count (`puschRxChEq` in [channel_eq.cu](../cuPHY/src/cuphy/channel_eq/channel_eq.cu)):

| `nRxAnt` | Layers supported | Kernel |
|---|---|---|
| 1–8 | 1–8 | `eqMmseCoefCompLowMimoKernel` |
| 16 | 1–8, 16 | `eqMmseCoefCompHighMimoKernel` (dispatches to `_v1` / `_v2` / `_v3`) |
| 64 | 16 | `eqMmseCoefCompMassiveMimoKernel_v1` |

| | Tensor | Type | Shape |
|---|---|---|---|
| **In** | `HEst` | C32F | `[nRxAnt, nLayers, NF, NH]` |
| **In** | `NoiseVarPreEq`, `LwInv` | R32F / C32F | see step 2 |
| **Out** | `EqCoef` (`tInfoEqCoef`) | C32F | `[nRxAnt, 12, nLayers, nPrb, NH]` |
| **Out** | `ReeDiagInv` (`tInfoReeDiagInv`) | R32F | `[12, nLayers, nPrb, nTimeChEq]`, where `nTimeChEq` = 4 with time-domain interpolation, else 1 |
| **Out** | `ChEqDbg` (debug only) | C32F | `[nLayers + nRxAnt, nLayers, NF, NH]` |

`ReeDiagInv` is the inverse of the post-equalizer error variance per layer and subcarrier; the soft demapper uses it to scale LLRs.

### Step 6 — Equalization and soft demapping (`channel_eq/`)

| Kernel | Used when |
|---|---|
| `eqMmseSoftDemapKernel` | default (≤ 16 antennas) |
| `eqMmseSoftDemapKernel_64R` | 64 receive antennas |
| `eqMmseEqualizationKernel` | `kernelSelOption` = `PUSCH_NO_SD_DERATE_MATCHING_FEC`: equalize only, no soft demapping |
| `eqMmseSoftDemap72eKernel` | O-RAN split 7.2e (`openRanFunctionalSplitOption`): equalized data and `ReeDiagInv` come from the fronthaul tensors `pTX_72e` / `pTReeInv_72e` |

Each kernel is compiled for a symbol bitmask: full slot `0x3FFF` (symbols 0–13) or early-HARQ `0xF` / `0x1F`.

| | Tensor | Type | Shape |
|---|---|---|---|
| **In** | `DataRx` | C16F | `[NF_cell, 14, nRxAnt]` |
| **In** | `EqCoef` | C32F | `[nRxAnt, 12, nLayers, nPrb, NH]` |
| **In** | `ReeDiagInv` | R32F | `[12, nLayers, nPrb, nTimeChEq]` |
| **In** | `CfoPhaseRot`, `TaPhaseRot` | C32F | see step 4 (when CFO correction is enabled) |
| **In** | QAM order per layer (`qam[]`) | — | from UE-group parameters |
| **Out** | `LLR` (`tInfoLLR`) | R16F (R32F if built without `LLR_FP16`) | `[8, nLayers, NF, Nd]`; first dimension sized for 256QAM (8 bits per symbol) |
| **Out** | `LLRCdm1` (`tInfoLLRCdm1`) | R16F | `[8, nLayers, NF, Nd]`; only when `nDmrsCdmGrpsNoData` = 1 |
| **Out** | `DataEq` (`tInfoDataEq`) | C16F | `[nLayers, NF, Nd]`; only with debug output, equalize-only mode, or split 7.2e |
| **Out** | `DataEqDft` (`tInfoDataEqDft`) | C32F | `[NF × Nd]`; equalized symbols for DFT-s-OFDM UEs (one UE and one layer per UE group), input to step 7 |

### Step 7 — DFT-s-OFDM (`channel_eq/`, when `enableDftSOfdm` = 1)

| Order | Kernel | Inputs | Outputs |
|---|---|---|---|
| (once, at object creation) | `bluestein_workspace_kernel` | — | Bluestein workspaces, time and frequency domain (C32F) |
| 7a | `bluestein_Idft_kernel` | `DataEqDft` `[NF × Nd]`, Bluestein workspaces | `DataEqDft`, overwritten in place with the IDFT result (one IDFT of size `NF` per data symbol) |
| 7b | `eqMmseSoftDemapAfterDftKernel` | `DataEqDft` (after IDFT), `ReeDiagInv` | `LLR` `[8, nLayers, NF, Nd]` |

Bluestein's algorithm is used because the IDFT size (12 × `nPrb`) is generally not a power of two.

### Measurements produced along the way

| Tensor | Type | Shape |
|---|---|---|
| `NoiseVarPostEq`, `SinrPostEq` | R32F | `[nUes]` |
| `SinrPreEq` | R32F | `[nUes]` |
| `Rsrp` | R32F | `[nUes]` |
| `Rssi` | R32F | `[nUeGrps]` |

---

## Part 2 — `channel_est/`: standalone 1-D MMSE estimator

`channel_est/` is a separate, older estimator. It is exposed as the public function `cuphyChannelEst1DTimeFrequency()` ([cuphy.h](../cuPHY/src/cuphy/cuphy.h), implemented in [cuphy.cpp](../cuPHY/src/cuphy/cuphy.cpp)) and used by the example [cuphy_ex_channel_est.cpp](../cuPHY/examples/channel_estimation/cuphy_ex_channel_est.cpp). The PUSCH receive pipeline does not call it.

### Call chain

```
cuphyChannelEst1DTimeFrequency()                 cuphy.cpp
  → channel_est::mmse_1D_time_frequency()        channel_est.cu
    → mmse_1D_time_frequency_launch()            channel_est.cu
      → mmse_1D_time_frequency_kernel            grid = nUEs blocks, 512 threads per block
```

The kernel applies a frequency-domain MMSE interpolation filter to the DMRS pilots, then a time-domain filter, producing an estimate for every subcarrier and all 14 symbols. Each thread block handles one UE. The kernel is written for 32 frequency pilots (`NFP` = 32) and 4 time pilots (`NTP` = 4) per UE, and 14 symbols.

### Tensors

| | Tensor | Type | Shape |
|---|---|---|---|
| **In** | `Y` (received symbols) | C32F or C16F | `[Nsc, Nsym, nAnt]` |
| **In** | `W_freq` (frequency filter) | R32F or R16F | rank 3: `[NscPerUE, NFP, nUEs]` |
| **In** | `W_time` (time filter) | R32F or R16F | rank 3, per UE |
| **In** | `DMRS_index_freq` | int16 | `[NFP, nUEs]` |
| **In** | `DMRS_index_time` | int16 | `[NTP, nUEs]` |
| **Out** | `Hinterp` (interpolated channel) | same type as `Y` | `[NscPerUE, Nsym, nAnt, nUEs]` |

The example derives `nUEs` from `DMRS_index_freq` dimension 1, `NscPerUE` from `W_freq` dimension 0, and `Nsym`, `nAnt` from `Y`.
