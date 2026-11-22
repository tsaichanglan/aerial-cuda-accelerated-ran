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
| 3 | RSRP measurement | `pusch_rssi/` | SINR measurement enabled. Runs here, before step 4, only when weighted-average CFO is also enabled; otherwise it runs after soft demapping (see step 3) |
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

*Example: one UE group with `nPrb` = 21, legacy MMSE.* 21 is ≥ 4 but not a multiple of 4, so the 4-in / 2-out kernel is used. It is also odd, so the odd-PRB handling above applies to the last block.

**Launch geometry.**

| | Value | Why |
|---|---|---|
| Blocks per antenna (`gridDim.x`) | `div_round_up(21, 2)` = **11** | 2 output PRBs per block; 11 × 2 = 22 would cover one PRB too many |
| `blockDim` | 48 | 2 output PRBs × 2 grids × 12 tones |
| `gridDim` (4 Rx antennas, 1 UE group) | `(11, 4, 1)` | 44 blocks, 2,112 threads |
| Launches | `NH` = `dmrsAddlnPos` + 1 | one per DMRS time position |

**Blocks.** In code terms, the window start is `prbClusterStartIdx` = 2*i* − 1, set to 0 for the first block and to `nPrb` − 4 = 17 for the last block ([ch_est.cu:612-614](../cuPHY/src/cuphy/ch_est/ch_est.cu#L612-L614)):

| Block (`blockIdx.x`) | Reads DMRS PRBs | Computes PRBs | Writes `HEst` PRBs | Filter |
|---|---|---|---|---|
| 0 | 0–3 | 0–1 | 0–1 | lower edge |
| 1 | 1–4 | 2–3 | 2–3 | middle |
| 2 | 3–6 | 4–5 | 4–5 | middle |
| *i* (middle) | 2*i*−1 … 2*i*+2 | 2*i* … 2*i*+1 | 2*i* … 2*i*+1 | middle |
| 9 | 17–20 | 18–19 | 18–19 | middle |
| 10 | 17–20 | 19–20 | **20 only** | upper edge |

- **Block 10 without the fix** would compute PRBs 20–21, but PRB 21 does not exist.
- **Shift:** because `nPrb` is odd, its output tone index is moved down by 12 tones ([ch_est.cu:658-659](../cuPHY/src/cuphy/ch_est/ch_est.cu#L658-L659)), so it computes PRBs 19–20.
- **Drop:** the threads with output tone < 12 (PRB 19) do not write ([ch_est.cu:1014](../cuPHY/src/cuphy/ch_est/ch_est.cu#L1014)), because block 9 already wrote PRB 19 with the middle filter.
- **Coverage:** each of PRBs 0–20 is written exactly once (10 blocks × 2 + 1 = 21).
- **Shared window:** blocks 9 and 10 read the same window, PRBs 17–20. Block 10 only applies the upper-edge filter rows to it.

**Tensors** (assuming 4 Rx antennas, 2 layers and one DMRS position):

| Tensor | Shape | Size |
|---|---|---|
| DMRS input read from `DataRx` | 21 PRBs × 12 subcarriers per antenna (126 per grid) | 252 REs per antenna |
| Shift sequence | `[21·6, 1]` = `[126, 1]` | |
| Unshift sequence | `[21·12·2 + 1]` = `[505]` | |
| `HEst` | `[nRxAnt, nLayers, NF, NH]` = `[4, 2, 252, 1]` | NF = 21 × 12 = 252; 2,016 complex values |

**Multi-stage algorithm (algorithm 1), kernel ①.** It uses fixed 4-PRB clusters, so grid x = `div_round_up(21, 4)` = 6. The clusters are PRBs 0–3, 4–7, …, 16–19, and then **PRB 20 alone**. In that last cluster, threads for PRBs above 20 are invalid (`lastPrbThisCluster` = 20, [ch_est.cu:1912](../cuPHY/src/cuphy/ch_est/ch_est.cu#L1912)), and its delay-correlation pairs stay inside PRB 20. With per-PRG estimation (12-PRB clusters), grid x = `div_round_up(21, 12)` = 2: PRBs 0–11 and 12–20.

**Sharing launches with other UE groups.** The kernel function depends on the PRB bracket (4-in / 2-out), not on the exact `nPrb`. A 21-PRB UE group can therefore share one launch with, for example, a 34-PRB group that has the same layer and DMRS configuration. That launch has grid x = max(11, 17) = 17, and the 21-PRB group's blocks 11–16 exit early.

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
    where `P` = 2 for one layer, or 4 when adjacent tones were first averaged. It writes τ to `DmrsDelayMean`, builds the shift and unshift ramps from τ, and applies steps 4–6 to `DmrsLSEst`. The per-PRG variant changes the clustering (12-PRB clusters, no correlation pairs across PRG edges) and the filtering, but still uses one τ per UE group. See [Multi-stage delay estimation](#multi-stage-delay-estimation-algorithm-1) below for the full math.
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

#### Multi-stage delay estimation (algorithm 1)

Algorithm 0 centres the channel with a fixed τ (0.4375 µs in the shipped `cuPhyChEstCoeffs.h5` at 30 kHz). Algorithm 1 measures τ per UE group from the pilots. Kernel ① accumulates a frequency correlation, and kernel ② turns it into a delay and builds the shift ramps from it.

**Kernel ① — accumulate the correlation** (`windowedChEstPreNoDftSOfdmKernel`, [ch_est.cu:2134-2238](../cuPHY/src/cuphy/ch_est/ch_est.cu#L2134-L2238))

Let `Ĥ_{t,r,ℓ}[m]` be the LS estimate after descrambling and OCC removal, for DMRS time position `t`, receive antenna `r`, layer `ℓ` and DMRS tone `m` of the layer's grid (subcarrier `2m + grid`). No shift is applied in kernel ①.

- **One layer in the UE group** (`nLayers` = 1): correlate neighbouring DMRS tones, 2 subcarriers apart (P = 2):

  ```
  R = Σ_t Σ_r Σ_c Σ_{m, m+1 ∈ c}  conj(Ĥ_{t,r,0}[m]) · Ĥ_{t,r,0}[m+1]
  ```

- **More than one layer** (`nLayers` > 1): fOCC removal in kernel ① only flips the sign of every other tone, so the other fOCC layer of the same grid is still present with alternating sign. Averaging each pair of adjacent tones cancels it. The pair averages are then correlated 2 DMRS tones apart, which is 4 subcarriers (P = 4):

  ```
  Ā_{t,r,ℓ}[m] = ½ (Ĥ_{t,r,ℓ}[m] + Ĥ_{t,r,ℓ}[m+1]),    m even
  R = Σ_t Σ_r Σ_ℓ Σ_c Σ_{m even, m..m+3 ∈ c}  conj(Ā_{t,r,ℓ}[m]) · Ā_{t,r,ℓ}[m+2]
  ```

`c` runs over the PRB clusters of the UE group: 4 PRBs normally (`CUPHY_PUSCH_RX_CH_EST_NON_PRG_DELAY_EST_PRB_CLUSTER_SIZE`), or 12 PRBs with per-PRG estimation. Pairs never cross a cluster boundary (`nextToneIsValid` / `nextToneIsValidAfterAvg`), and with per-PRG estimation they also never cross a PRG boundary (`isValidR1Cal`).

Each thread block reduces its partial sum (warp `reduce`, then shared memory), and thread 0 adds it to the UE group's accumulator with `atomicAdd(&DmrsAccum[dmrsActiveAccumBuf], …)`. Block (0, 0) clears the other buffer, and setup flips `dmrsActiveAccumBuf` every slot ([ch_est.cu:6748](../cuPHY/src/cuphy/ch_est/ch_est.cu#L6748)), so the next slot starts from zero.

**Kernel ② — convert to a delay** (`chEstDelayShiftReduction`, [ch_est.cu:1240-1274](../cuPHY/src/cuphy/ch_est/ch_est.cu#L1240-L1274))

For a channel `H[k] = Σ_p α_p · e^{−j2π·k·Δf·τ_p}`, the correlation P subcarriers apart is `E{conj(H[k])·H[k+P]} = Σ_p |α_p|² · e^{−j2π·P·Δf·τ_p}`. Its phase is therefore set by the power-weighted delay of the taps:

```
τ̂ = − arg(R) / (2π · P · Δf),     Δf = 15 kHz · 2^μ,     P = 2 (nLayers = 1) or 4 (nLayers > 1)
```

In code, `atan2f(R.y, R.x)` gives `arg(R)`, the value in units of `1/Δf` is stored in `DmrsDelayMean[t]`, and the result is then divided by `df` to get seconds. Since `arg(R)` lies in (−π, π], τ̂ is unambiguous for `|τ| < 1/(2·P·Δf)`:

| Δf | P = 2 | P = 4 |
|---|---|---|
| 15 kHz | ±16.7 µs | ±8.3 µs |
| 30 kHz | ±8.3 µs | ±4.2 µs |

Kernel ② then builds the ramps in shared memory, using the window size `nPrbIn` (= `nPrb` if `nPrb` ≤ 3; 8 if `nPrb` ≥ 8 and `nPrb` % 4 = 0; otherwise 4):

```
shiftSeq[i]   = e^{+j2π·Δf·τ̂·(2i)},      i = 0 … 6·nPrbIn − 1      (DMRS tones)
unShiftSeq[i] = e^{−j2π·Δf·τ̂·(i − 1)},   i = 0 … 12·nPrbIn        (all subcarriers, from −1)
```

These replace the static `ShiftSeq` / `UnShiftSeq` tables of algorithm 0. The static MMSE filters `WFreq` are then applied to `DmrsLSEst` as in steps 4–6.

**More than one DMRS symbol.** τ̂ is **not** computed from the first DMRS symbol only:

- Within a DMRS position, a double-symbol DMRS (`dmrsMaxLen` = 2) is first combined by tOCC removal, so both symbols contribute to `Ĥ`.
- Across DMRS positions, kernel ① runs once per position `t` = 0 … `dmrsAddlnPos`, and every launch adds to the **same** accumulator. All kernel ① launches finish before any kernel ② starts: the stream order is ① for all `t`, then ② for all `t`, and in a graph the ② nodes depend on all ① nodes.
- Every kernel ② launch therefore reads the same `R` and uses the same τ̂ for every DMRS position. All entries of `DmrsDelayMean[t]` are equal.
- **Exception, early HARQ:** the sub-slot graph runs ① and ② for `t` = 0 as soon as the first DMRS position arrives, so that early `HEst` uses only the first position. The full-slot pass skips ① for `t` = 0, because its sum is already in the accumulator, adds the remaining positions, and reruns ② for every `t`, including `t` = 0, with the full-slot τ̂ ([ch_est_stream.cpp:56-70](../cuPHY/src/cuphy/ch_est/ch_est_stream.cpp#L56-L70)).

**PRB range and MU-MIMO.** There is one accumulator per UE group (`DmrsAccum` is `[2]` per UE group, [pusch_rx.cpp:10695](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L10695)). It sums over all layers of all UEs in the group, all receive antennas, all DMRS positions and the group's **whole** PRB allocation `startPrb … startPrb + nPrb − 1`. τ̂ is one value per UE group, not per UE.

*Example:* a MU-MIMO UE group of 100 PRBs with UE A on ports 0/1 (grid 0, fOCC 0/1) and UE B on ports 2/3 (grid 1, fOCC 0/1), single-symbol DMRS, no additional position:

| Item | Value |
|---|---|
| `nLayers` (group total) | 4, so the averaged path is used and P = 4 |
| Kernel ① grid / block | `(100 / 4 = 25, nRxAnt, 1)` / 4 × 12 = 48 threads |
| PRBs used | all 100 PRBs, for both UEs |
| Pairs per layer, cluster and antenna | grid tones `m` = 0 … 23; `m` even with `m + 3` ≤ 23 gives `m` = 0, 2, …, 20 = 11 pairs |
| Terms in `R` | 25 clusters × 11 pairs × 4 layers × `nRxAnt` (each grid-0 thread adds UE A's layers, each grid-1 thread adds UE B's) |
| Result | one τ̂, written to `DmrsDelayMean[0]`, used to centre `HEst` for all four layers |

Because each term is weighted by `|H|²`, the stronger UE dominates τ̂. If the two UEs have different timing offsets, τ̂ is a power-weighted compromise: the weaker UE's channel is centred less accurately before filtering, which raises its estimation error. Per-UE timing offsets are measured later, in step 4 (`cfoTaEst`), but they do not feed back into the channel-estimate shift.

### Step 2 — Noise and interference estimation (`pusch_noise_intf_est/`)

| Kernel | Inputs | Outputs |
|---|---|---|
| `noiseIntfEstNoDftSOfdmKernel` / `noiseIntfEstKernel` | `DataRx` (DMRS symbols), `HEst` | `NoiseVarPreEq`, `LwInv` |

| Tensor | Type | Shape | Notes |
|---|---|---|---|
| `NoiseVarPreEq` (`tInfoNoiseVarPreEq`) | R32F | `[nUes]` (`[nUeGrps]` in some configurations) | pre-equalizer noise-plus-interference power |
| `LwInv` (`tInfoLwInv`) | C32F | `[nRxAnt, nRxAnt, nPrb]` | inverse Cholesky factor of the per-PRB noise-interference covariance; allocated only for MMSE-type equalizers or SINR measurement |

#### What noise and interference estimation calculates

The idea: rebuild the noise-free DMRS signal from the channel estimate, subtract it from what was received, and measure the power and spatial covariance of what is left over. Code: `noiseIntfEstNoDftSOfdmKernelInner`, [pusch_noise_intf_est.cu:493-1129](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L493-L1129). The DFT-s-OFDM kernel `noiseIntfEstKernel` follows the same steps, with low-PAPR sequences in place of the Gold sequence.

**Launch.** Grid `(⌈nMaxPrb / N_PRB_PER_THRD_BLK⌉, nUeGrps)`, block `(12 · nRxAnt, N_PRB_PER_THRD_BLK)`, where `N_PRB_PER_THRD_BLK` is 1 or 2 (always 2 for DFT-s-OFDM). There is one thread per (subcarrier, receive antenna) of a PRB, and `threadIdx.y` selects the PRB. Blocks beyond a UE group's own `nPrb` exit early.

**1 · Regenerate the transmitted DMRS** for each layer `ℓ`, subcarrier `k` and DMRS symbol `s`, but only on the grid (CDM group) that the layer's port uses:

```
x_ℓ[k, s] = β · w_f,ℓ[k] · w_t,ℓ[s] · c[k, s]

c        Gold-sequence QPSK symbol (±1 ± j)/√2, seeded from dmrsScrmId, scid, slot, symbol
w_f,ℓ    fOCC: −1 on odd grid tones if the port's bit 0 is set, otherwise +1
w_t,ℓ    tOCC: −1 on the 2nd symbol of a double-symbol DMRS if the port's bit 2 is set, otherwise +1
β        √2 if nDmrsCdmGrpsNoData = 2 (DMRS power boost), otherwise 1
x_ℓ = 0  on the grid the port does not use (port bit 1 selects the grid)
```

**2 · Residual** (noise, interference and channel-estimation error), per receive antenna `r`:

```
e_r[k, s] = Y_r[k, s] − Σ_ℓ Ĥ[r, ℓ, k, t(s)] · x_ℓ[k, s]
```

`Ĥ` is `HEst` for the DMRS position `t(s)` that contains symbol `s` (`t = s` for single-symbol DMRS, `t = ⌊s/2⌋` for double-symbol). With `nDmrsCdmGrpsNoData` = 2, a grid with no layers of this UE group has `x = 0`, so its residual is the received signal there: pure noise plus interference.

**3 · Per-PRB covariance.** For each PRB `p`, with `e[k, s] = [e_0 … e_{nRxAnt−1}]ᵀ`:

```
R_ww[p] = (1 / N) · Σ_{s ∈ DMRS symbols} Σ_{k ∈ K_p} e[k, s] · e[k, s]ᴴ  +  ε·I

nDmrsCdmGrpsNoData = 2:  K_p = all 12 subcarriers of PRB p,        N = 12 · nDmrsSyms
nDmrsCdmGrpsNoData = 1:  K_p = the 6 even subcarriers (grid 0),    N = 6 · nDmrsSyms
ε = CUPHY_NOISE_REGULARIZER = 1e-8
```

With one CDM group without data, the odd subcarriers carry PUSCH data, so they cannot be used. Only the diagonal (per-antenna power) is computed unless `eqCoeffAlgo` is an MMSE-IRC type, which needs the full matrix.

**4 · Shrinkage** (only for `eqCoeffAlgo` = `MMSE_IRC_SHRINK_RBLW` or `MMSE_IRC_SHRINK_OAS`). With `T` = N real samples and `A = nRxAnt`:

```
R_ww[p] ← (1 − ρ) · R_ww[p] + ρ · (tr(R_ww[p]) / A) · I

RBLW:  ρ = clip₀¹( ((T−2)/T · tr(R²) + tr(R)²) / ((T+2) · (tr(R²) − tr(R)²/A)) )
OAS:   ρ = clip₀¹( ((1 − 2/A) · tr(R²) + tr(R)²) / ((T + 1 − 2/A) · (tr(R²) − tr(R)²/A)) )
```

**5 · Whitening factor** (MMSE-IRC types only): Cholesky factorization `R_ww[p] = L_p · L_pᴴ`, then `LwInv[:, :, p] = L_p⁻¹` (`cholFactorInv`). The equalizer coefficient kernel uses it to whiten the received signal and the channel per PRB.

##### How L_p⁻¹ is computed (`cholFactorInv`)

Code: [pusch_noise_intf_est.cu:317-484](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L317-L484). It runs once per PRB on R_ww[p], after ε·I has been added and shrinkage applied. The work happens in place in shared memory, with A = nRxAnt threads per PRB. A block's PRBs are processed one after another ([:1061-1065](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L1061-L1065)).

**L is computed explicitly, then inverted.** There is no direct R_ww → L⁻¹ algorithm. Two passes share one buffer: a standard Cholesky factorization, then a triangular inversion. Only L⁻¹ is written to global memory (`LwInv`). L exists only temporarily in shared memory.

**Pass A: Cholesky factorization R_ww = L·Lᴴ** (`cholFactor`, [:317-378](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L317-L378)). It works one column j at a time:

```
L_jj = sqrt( R_jj − Σ_{k<j} |L_jk|² )                       thread 0                       (:336-342)
L_ij = ( R_ij − Σ_{k<j} L_ik · conj(L_jk) ) · (1/L_jj),     i > j, thread i, in parallel    (:358-367)
```

Two storage tricks prepare for pass B:

- **The diagonal holds 1/L_jj, not L_jj** ([:346](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L346)). The division is done once. For a triangular matrix, 1/L_jj is also the diagonal of L⁻¹, so the inverse's diagonal is already finished after pass A.
- **Each L_ij is mirrored into the upper triangle**, `L(j,i) = L(i,j)` ([:370](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L370)). Pass B can then overwrite the lower triangle with L⁻¹ while it still reads L from the upper copy.

**Pass B: triangular inverse X = L⁻¹** (`triLInv`, [:386-434](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L386-L434)). It solves L·X = I. Thread j computes column j from top to bottom:

```
X_jj = 1/L_jj                                    (already stored by pass A)
X_ij = −(1/L_ii) · Σ_{k=j}^{i−1} L_ik · X_kj,    i > j                             (:414-427)
```

L_ik is read from the mirrored upper copy, and X_kj from earlier rows of the same column in the lower triangle. All A columns are computed in parallel with no further synchronization. As X_ij is written, it overwrites L_ij in the lower triangle.

**Pass C: write-out** ([:472-481](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L472-L481)). The lower triangle (X) goes to `LwInv[:, :, p]`, and the upper triangle is set to 0.

**2×2 check.** For R = [[a, conj(b)], [b, c]]:

```
L₁₁ = √a,   L₂₁ = b/√a,   L₂₂ = √(c − |b|²/a)

L⁻¹ = [  1/L₁₁                  0     ]
      [ −L₂₁/(L₁₁·L₂₂)        1/L₂₂   ]
```

**No guard on the square root.** If rounding made R_jj − Σ|L_jk|² negative, `sqrtf` would return NaN. The only protection is the ε = 1e-8 diagonal load from point 3, plus the shrinkage in point 4 for algorithms 3 and 4. Compare the LDL factorization of G in step 5, whose pivots are always ≥ 1.

**6 · Scalar noise variance per UE group.** Each block adds `Σ_p tr(R_ww[p])` for its PRBs into the output with `atomicAdd`. The last block to finish (counted with `interCtaSyncCnt`) then computes:

```
σ²_w = (1 / (nRxAnt · nPrb)) · Σ_{p=0}^{nPrb−1} tr(R_ww[p])        (average per antenna, per resource element)

NoiseVarPreEq  = 10·log10(σ²_w) + 0.5 dB      (correction for noise removed by the interpolation filter)
invNoiseVarLin = 1 / σ²_w                     (used by MMSE equalizer coefficients)
noiseVarForDtx = σ²_w                         (used by DTX detection)
```

With `ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR` (ON by default), the sum is accumulated at the first UE's index and the same dB value is then written for every UE in the group. There is no separate per-UE estimate.

**More than one DMRS symbol.** The full-slot kernel (`CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT`) uses **all** DMRS symbols: `s` runs over `nDmrsSyms` = (`dmrsAddlnPos` + 1) × `dmrsMaxLen`, each symbol is compared with `HEst` of its own DMRS position, and all symbols are averaged into one `R_ww[p]` (the `1/N` above). Only the early-HARQ variant (`CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0`) restricts `s` to the first DMRS position (`nDmrsSyms = dmrsMaxLen`), so that early results are available before the remaining DMRS symbols arrive.

**PRB range.**

| Output | Granularity | PRBs used |
|---|---|---|
| `R_ww[p]`, `LwInv[:, :, p]` | one per PRB | the 12 (or 6) subcarriers of that PRB only, across all DMRS symbols |
| `NoiseVarPreEq`, `invNoiseVarLin` | one per UE group | the group's whole allocation `startPrb … startPrb + nPrb − 1`, and all receive antennas |

For a MU-MIMO UE group, the reconstruction in step 2 subtracts the layers of **all** UEs in the group. The residual, the covariance and the scalar variance are therefore shared by all those UEs. Signals from other UE groups or other cells on the same PRBs are not subtracted, so they count as interference.

### Step 3 — RSRP (optional, `pusch_rssi/`)

Uses `HEst`; writes `Rsrp` (R32F, `[nUes]`). RSSI (`[nUeGrps]`, plus a per-symbol/per-antenna `RssiFull` of `[14, 64, nUeGrps]`) is measured from `DataRx` in the same directory.

#### What RSRP measurement calculates

RSRP is the average power of the channel estimate `HEst` at the DMRS positions. It is not measured on the received samples. The same kernel also turns RSRP into the pre-equalizer SINR, and in one variant into the post-equalizer SINR.

**When it runs and where it sits in the graph.** It runs only if `enableSinrMeasurement` = 1 ([pusch_rx.cpp:2718](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L2718)). The kernel and its position depend on `enableWeightedAverageCfo` ([pusch_rssi.cu:1805-1878](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L1805-L1878)):

| `enableWeightedAverageCfo` | Kernel | Graph position | Outputs |
|---|---|---|---|
| 1 | `rsrpMeasWithoutPostEqMeasKernel_v1` | Front-end chain channel estimation → noise → **RSRP** → CFO → equalizer coefficients ([pusch_rx.cpp:4246-4310](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L4246-L4310)), because step 4 needs `SinrPreEq` | `Rsrp`, `SinrPreEq` |
| 0 | `rsrpMeasKernel_v1` | After soft demapping, beside the RSSI and UCI nodes ([pusch_rx.cpp:4413-4426](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L4413-L4426)), because it also reads `ReeDiagInv` from step 5 | `Rsrp`, `SinrPreEq`, `SinrPostEq`, `NoiseVarPostEq` |
| (early HARQ) | `sinrMeasEarlyHarqKernel_v1` | Sub-slot graph, after early-HARQ soft demapping | `RsrpEhq`, `SinrPreEq` |

`rsrpMeasKernel_v2` ([pusch_rssi.cu:1419](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L1419)) computes RSRP only. The current setup does not select it.

**Launch** ([pusch_rssi.cu:1745-1783](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L1745-L1783)):
- grid = (⌈12·nMaxPrb / 48⌉, nUeGrps, 1), and at least enough threads to cover `maxLayers·nRxAnt`;
- block = (32, 4) = 128 threads;
- shared memory holds 2 floats per layer.

Each thread block walks through the group's flattened `HEst` array with a grid stride that is a multiple of `nRxAnt·nLayers`. Each thread therefore always sees the same layer, so it can accumulate locally. The power is then reduced per layer with a warp `cg::reduce` and an `atomicAdd` into shared memory. Finally, one global `atomicAdd` per layer goes into `Rsrp[absUeIdx]`. The last block to finish for the group, found with the `interCtaSyncCnt` counter, does the final normalization.

**Notation.** Ĥ[r, ℓ, k, t] = `HEst` at receive antenna r, layer ℓ, subcarrier k (0 … 12·nPrb − 1, relative to the group's `startPrb`) and DMRS position t (0 … nTimeChEst − 1, with nTimeChEst = `dmrsAddlnPos` + 1). L_u is the layer count of UE u, and ℓ ∈ u means layer ℓ belongs to UE u through `ueGrpLayerToUeIdx`.

**1 · RSRP per UE** ([pusch_rssi.cu:857-902, 944-1000](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L857-L1000)):

```
P_u      = Σ_{ℓ∈u} Σ_t Σ_k Σ_r |Ĥ[r, ℓ, k, t]|²          (accumulated linearly in Rsrp[absUeIdx])

RSRP_lin = P_u / (12·nPrb · nRxAnt · nTimeChEst)
Rsrp     = 10·log10(RSRP_lin)   [dB]                    (overwrites the linear sum in place)
```

- **Averaged** over subcarriers, receive antennas and DMRS positions.
- **Summed over the UE's layers**, not averaged ("We don't average across layers", [:995](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L995)). A 2-layer UE with equal per-layer power therefore reports 3 dB more than a 1-layer UE.
- The value is in the digital scale of the received IQ samples. No dBm calibration or Rx-gain offset is applied in this kernel.
- `HEst` is the channel seen by unit-amplitude DMRS. The DMRS power boost β (√2 when `nDmrsCdmGrpsNoData` = 2) is removed during channel estimation, so it is not part of the RSRP.

**2 · Pre-equalizer SINR per UE** ([:1002-1011](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L1002-L1011)):

```
SinrPreEq_dB = Rsrp_dB − NoiseVarPreEq_dB
```

`NoiseVarPreEq` comes from step 2. With `USE_PUSCH_PER_UE_PREQ_NOISE_VAR` = 1 (the default) it is read at `absUeIdx`, otherwise at `ueGrpIdx`. In both cases it holds the same group-wide value, so in a MU-MIMO group the UEs differ only through their own RSRP. This SINR is the per-slot weight used by the weighted-average CFO in step 4.

**3 · Post-equalizer SINR per UE** (`rsrpMeasKernel_v1` only, [:905-950, 1013-1025](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L905-L1025)):

`ReeDiagInv[k, ℓ, prb, t]` = 1/Ree is the inverse of the per-layer MMSE error variance written by the step 5 coefficient kernels. It is capped at `MAX_ERROR_PRECISION` ([channel_eq.cu:876-880](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L876-L880)), and it is the post-equalizer SINR of each resource element.

```
t*               = nTimeChEst − 1 if enablePuschTdi, else 0      (one time instance only)
A_u              = Σ_{ℓ∈u} Σ_k ReeDiagInv[k, ℓ, t*]                (accumulated in SinrPostEq[absUeIdx])

NoiseVarPostEq_lin = 12·nPrb · L_u / A_u                           (harmonic mean of Ree)
NoiseVarPostEq     = 10·log10(NoiseVarPostEq_lin)   [dB]
SinrPostEq         = −NoiseVarPostEq                [dB]
```

Averaging 1/Ree and inverting gives the harmonic mean of the error variance, so a few bad subcarriers lower `SinrPostEq` less than an arithmetic mean of Ree would.

**More than one DMRS symbol.** The full-slot RSRP uses **all** DMRS positions t = 0 … `dmrsAddlnPos` and divides by `nTimeChEst`. `HEst` holds one estimate per DMRS position, and with double-symbol DMRS the two symbols share that estimate. Post-equalizer SINR uses only one time instance, `t*`.

With early HARQ, which uses only the first DMRS position:

| Step | Kernel | DMRS positions | What it does |
|---|---|---|---|
| 1 | `sinrMeasEarlyHarqKernel_v1` | t = 0 only (`nTimeChEst` = 1) | Divides by 1 and writes dB to `RsrpEhq` and `SinrPreEq`. It leaves the linear t = 0 sum in `Rsrp`, then resets `NoiseVarPreEq` and the sync counter ([pusch_rssi.cu:1375-1403](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L1375-L1403)). |
| 2 | Full-slot kernel, `CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS` | t ≥ 1 (starts at t = 1, [:868-871](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L868-L871)) | Adds onto the t = 0 sum and divides by the full `nTimeChEst`. The final `Rsrp` therefore still averages every DMRS position, and t = 0 is not counted twice. |

With `subSlotProcessingFrontLoadedDmrsEnabled`, both paths use `CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS` ([pusch_rx.cpp:2727, 2741](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L2727)).

**PRB range.**

| Output | Granularity | PRBs used |
|---|---|---|
| `Rsrp`, `SinrPreEq` | one per UE | the UE group's whole allocation (12·nPrb subcarriers of `HEst`), all receive antennas, **only that UE's layers** |
| `SinrPostEq`, `NoiseVarPostEq` | one per UE | the whole allocation, that UE's layers, one time instance |
| `NoiseVarPreEq` (input) | one per UE group | shared by all UEs of the group (step 2) |

For a MU-MIMO group, for example 100 PRBs shared by UE 0 (layers 0–1) and UE 1 (layer 2), the grid stride is a multiple of `nRxAnt·nLayers`. Each thread's layer therefore stays fixed, and `ueGrpLayerToUeIdx` adds layers 0–1 into `Rsrp[UE 0]` and layer 2 into `Rsrp[UE 1]`. Both UEs average over all 1200 subcarriers and all antennas. Because the layers of the two UEs are separated during channel estimation (by OCC and CDM group), each UE's RSRP contains only its own ports' power.

**Outputs to L2.** `Rsrp`, `SinrPreEq` and, when computed, `SinrPostEq` and `NoiseVarPostEq` are copied to the host with the other measurements ([pusch_rx.cpp:8391-8395](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L8391-L8395)) for the FAPI UL indications.

### Step 4 — CFO and timing-offset estimation (`cfo_ta_est/`)

| Kernel | Inputs | Outputs |
|---|---|---|
| `cfoTaEstLowMimoKernel` | `HEst` (estimates at different DMRS times) | `CfoEst`, `CfoPhaseRot`, `TaPhaseRot`, `CfoHz`, `TaEst` |

| Tensor | Type | Shape | Notes |
|---|---|---|---|
| `CfoEst` (`tInfoCfoEst`) | C32F | `[14, nUes]` | per-symbol CFO **compensation** factor per UE; the only CFO tensor the equalizer reads |
| `CfoPhaseRot` | C32F | `[4, 8, maxUeGrps]` (`[NH max, layers max, UE groups max]`) | internal accumulator: per-layer time correlation `R_t(ℓ)` between consecutive DMRS positions |
| `TaPhaseRot` | C32F | `[8, nUeGrps]` | internal accumulator: per-layer frequency correlation `F(ℓ)` between adjacent subcarriers |
| `CfoHz`, `TaEst` | R32F | `[nUes]` | reported CFO (Hz) and timing offset (µs); `TaEst` is copied to the host as `pTaEsts` when `enableToEstimation` is set |

#### What CFO and timing-offset estimation calculates

Code: `cfoTaEstLowMimoKernel_v2`, [cfo_ta_est.cu:262-648](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L262-L648). Both estimates come from phase rotations of `HEst`. The CFO is the phase change **in time** between DMRS positions, and the timing offset is the phase slope **in frequency** across subcarriers.

**When it runs.** It runs only if the UE group has an additional DMRS position. Setup sets `enableCfoCorrection = enableCfoCorrection(static) && dmrsAddlnPos > 0`, and the same for `enableToEstimation` ([pusch_rx.cpp:11989-11990](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L11989-L11990)). Kernels exist for `N_TIME_CH_EST` = 2, 3 or 4 (`dmrsAddlnPos` = 1…3) and 1–8 receive antennas. With a single DMRS position, no CFO is estimated or corrected, and no TA is reported, even though one position would be enough for the frequency slope.

**Launch.** Grid `(⌈nMaxPrb / 8⌉, nUeGrps)`, block `(⌈12·nRxAnt / 32⌉·32, nLayers)`. Each block covers 8 PRBs, and each `threadIdx.y` is one layer, with one thread per (subcarrier, antenna) pair. Partial sums are reduced within warps, then in shared memory, then across blocks with `atomicAdd` into `CfoPhaseRot` / `TaPhaseRot`. The last block of each UE group (counted with `interCtaSyncCnt`) computes the final values.

Notation: `Ĥ[r, ℓ, k, t]` = `HEst` for antenna `r`, layer `ℓ`, subcarrier `k` and DMRS position `t` = 0 … `nCHEST − 1`, with `nCHEST = dmrsAddlnPos + 1`. `l_t` = OFDM symbol index of the first symbol of DMRS position `t`, which is `dmrsSymLoc[t · dmrsMaxLen]`. `u` = UE, and `L_u` = number of layers of UE `u` (`nUeLayers`).

**1 · CFO: time correlation per layer and per pair of consecutive DMRS positions**

```
R_t(ℓ) = Σ_{k ∈ all nPrb·12 subcarriers} Σ_r  Ĥ[r, ℓ, k, t+1] · conj(Ĥ[r, ℓ, k, t]),      t = 0 … nCHEST − 2

θ_t(ℓ) = arg(R_t(ℓ)) / (l_{t+1} − l_t)          phase advance per OFDM symbol [rad / symbol]
```

The angles, not the correlations, are then averaged over the UE's layers and the DMRS pairs:

```
θ̄_u = (1 / (L_u · (nCHEST − 1))) · Σ_{ℓ ∈ u} Σ_t θ_t(ℓ)
```

The code accumulates `φ = −θ` (`shAccumCfoPhase`), so the stored value is already the correction angle.

##### How `shAccumCfoPhase` is accumulated and used

`shAccumCfoPhase` is the per-UE running sum from which `CfoEst` and `CfoHz` are produced ([cfo_ta_est.cu:546-620](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L546-L620)). The steps below run only in the last thread block to finish for the UE group.

**(a) It holds the negative angle: the correction, not the measurement.** Each thread takes one (layer ℓ, DMRS pair t) and computes ([:559](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L559)):

```
cfoPhase = −atan2(Im R_t(ℓ), Re R_t(ℓ)) / (l_{t+1} − l_t)  =  −θ_t(ℓ)    [rad / OFDM symbol]
```

If a positive CFO rotates the channel by e^{+jθ} per symbol, then arg R = +θ·(l_{t+1} − l_t). The negated value φ = −θ is the angle that undoes the rotation, so later steps use it directly in e^{jφ(…)} without flipping the sign. The symbol index is `pPilotSymbPos[t·dmrsMaxLen]` ([:554-555](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L554-L555)). With double-symbol DMRS, the spacing is therefore measured between the first symbols of the DMRS groups.

**(b) It is a per-UE sum, not a mean.**

```cpp
uint8_t ueIdx = pUeGrpLayerToUeIdx[layerIdx1];
atomicAdd(&shAccumCfoPhase[ueIdx], cfoPhase);      // :560-562
```

- There is one shared-memory slot per UE. The array is sized `N_LAYERS`, which is safe because a group never has more UEs than layers ([:318](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L318)).
- Every layer of UE u adds its value for every DMRS pair, which gives L_u·(nCHEST − 1) terms.
- In a MU-MIMO group, `ueGrpLayerToUeIdx` routes each layer to its own UE's slot, so the UEs' CFOs never mix.
- No division happens at this point:

```
S_u = Σ_{ℓ∈u} Σ_{t=0}^{nCHEST−2} (−θ_t(ℓ))
```

**(c) Readers divide by L_u·(nCHEST − 1)** to get the mean φ̄_u = −θ̄_u:

| Use | Code | Formula |
|---|---|---|
| Per-symbol compensation factor | [:613-616](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L613-L616) | `CfoEst[s,u]` = e^{j·S_u·(s − l₀)/(L_u(nCHEST−1))} = e^{−jθ̄_u(s − l₀)} |
| Reported CFO in Hz | [:618-620](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L618-L620) | taken at s − l₀ = 1, so `cfoAngle` = −θ̄_u and `CfoHz` = −cfoAngle·scs·10⁶/(2π·15·71.35) = θ̄_u/(2π·T_sym) |

l₀ = `pPilotSymbPos[0]`, so `CfoEst` = 1 on the first DMRS symbol, which is the reference phase all estimates are aligned to.

**(d) The weighted-average path writes back in the same summed form.** When `enableWeightedAverageCfo` is set ([:586-597](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L586-L597)):

```
θ_curr = −S_u / (L_u(nCHEST−1))                  // back to a positive mean angle
w      = (1 − α)·nPrb·10^{SINR_dB/10}
b     ← α·b + w·e^{jθ_curr}                       // per-UE buffer kept across slots (pFoCompensationBuffers)
S_u   ← −arg(b) · L_u(nCHEST−1)                   // re-negate and re-scale
```

The final multiplication by L_u(nCHEST − 1) exists only so that the unchanged divide-by-count code in (c) still works. This cross-slot filter averages phasors, so it handles the ±π wrap correctly.

**Worked example.** 30 kHz SCS, DMRS on symbols 2 and 11 (nCHEST = 2), and a 2-layer UE:

1. **Measured angles:** arg R is 0.090 rad on layer 0 and 0.108 rad on layer 1, over a spacing of 9 symbols. So θ = 0.010 and 0.012 rad/symbol.
2. **Sum:** `shAccumCfoPhase[u]` = −0.010 − 0.012 = −0.022.
3. **Mean:** φ̄ = −0.022/2 = −0.011.
4. **Compensation factor:** `CfoEst[s]` = e^{−j0.011(s−2)}. It is 1 at s = 2, has angle −0.099 rad at s = 11, which exactly cancels the rotation measured on the second DMRS, and has angle −0.121 rad at s = 13.
5. **Reported CFO:** T_sym = 35.68 µs, so `CfoHz` = 0.011/(2π·35.68 µs) ≈ 49 Hz.

**Caveats from the code as written.**
- **Angles, not phasors, are averaged within a slot.** If layers measure angles near ±π (for example +3.1 and −3.1 rad), the mean is about 0 instead of about π. This matters only near the unambiguous limit, about ±1.56 kHz for DMRS on symbols 2 and 11. The cross-slot filter in (d) does not have this problem.
- **The order of the float `atomicAdd`s is not fixed**, so the sum can differ in the last bits from run to run.
- **The divisor assumes every layer contributes every DMRS pair.** This holds because all threads with `thrdIdx < nLayers·(nCHEST−1)` take part.

**2 · CFO compensation factor per data symbol** (`CfoEst`, for all 14 symbols `s`):

```
CfoEst[s, u] = e^{ −j · θ̄_u · (s − l_0) }
```

The reference is the first DMRS symbol `l_0`: there the factor is 1, and every other symbol is rotated back to the phase of `l_0`.

**3 · CFO in Hz** (reported as `CfoHz`):

```
CFO_Hz = θ̄_u / (2π · T_sym),      T_sym = 71.35 µs · 15 / scsKHz
```

71.35 µs is the 15 kHz OFDM symbol length including a normal CP, (2048 + 144) / 30.72 MHz. The estimate is unambiguous while `|θ̄ · (l_{t+1} − l_t)| < π`, that is `|CFO| < 1 / (2 · Δl · T_sym)`. For example, with DMRS on symbols 2 and 11 at 30 kHz (Δl = 9, T_sym = 35.7 µs), the limit is about ±1.56 kHz.

**Optional weighted average across slots** (`enableWeightedAverageCfo`, which requires CFO correction). Each UE keeps a persistent two-float IIR state `b` (`pFoCompensationBuffers[ueIdx]`). With forgetting factor `α = foForgetCoeff`:

```
b ← α · b + (1 − α) · nPrb · 10^{SINR_preEq,dB / 10} · e^{ jθ̄_u }
θ̄_u ← arg(b)
```

Slots with more PRBs or a better pre-equalizer SINR therefore carry more weight. Because this needs `SinrPreEq`, the graph becomes a chain: channel estimation → noise → RSRP/SINR → CFO → equalizer coefficients.

**4 · Timing offset: frequency correlation per layer.** Only adjacent subcarriers inside the same PRB are paired (`k` = 0…10 within each PRB), summed over all PRBs, antennas and **all** DMRS positions:

```
F(ℓ) = Σ_t Σ_{p=0}^{nPrb−1} Σ_r Σ_{k=0}^{10}  Ĥ[r, ℓ, 12p+k+1, t] · conj(Ĥ[r, ℓ, 12p+k, t])

ψ̄_u = (1 / L_u) · Σ_{ℓ ∈ u} arg(F(ℓ))

TaEst_u [µs] = − ψ̄_u / (2π · Δf),      Δf = scsKHz · 1000 Hz
```

A delay τ appears as `H[k] ∝ e^{−j2π·k·Δf·τ}`, hence the minus sign. The range is `|TA| < 1 / (2·Δf)`, that is ±16.7 µs at 30 kHz. `HEst` has already been unshifted in step 1, so it contains the true delay, and the result does not depend on the delay centring of algorithm 0 or 1.

**PRB range and antennas.** Every sum runs over the UE group's whole allocation (`startPrb … startPrb + nPrb − 1`) and all `nRxAnt` antennas. The results are **per UE**, because only that UE's layers (`ueGrpLayerToUeIdx`) enter `θ̄_u` and `ψ̄_u`. In a MU-MIMO group, each UE gets its own CFO and TA from its own layers' channel estimates, over the shared PRBs.

**How the CFO is compensated** (in `channel_eq/`, only when `enableCfoCorrection` = 1):

1. **Equalizer coefficients (step 5).** The channel estimate of each DMRS position after the first is rotated back to the phase of `l_0` before the coefficients are computed:
   ```
   H_comp[:, ℓ, k, t] = Ĥ[:, ℓ, k, t] · CfoEst[l_t, u(ℓ)],      t > 0
   ```
   This applies in `eqLegacyMmseCoefCompLowMimoKernel` ([channel_eq.cu:2446-2451](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L2446-L2451)) and `eqMmseIrcCoefCompLowMimoKernel` ([:3043-3044](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3043-L3044)). After this, all `NH` estimates share the phase reference `l_0`, so time interpolation and per-position coefficients do not mix in CFO rotation.
2. **Equalized symbols (step 6).** Each equalized symbol is rotated by its symbol's factor:
   ```
   x̂_comp[ℓ, k, s] = x̂[ℓ, k, s] · CfoEst[s, u(ℓ)]
   ```
   This applies in `eqMmseSoftDemapKernel_v4` ([channel_eq.cu:3851-3855](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3851-L3855)), before LLR computation, and in the equalize-only `eqMmseEqualizationKernel_v4` ([:4149](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L4149)). It removes the remaining rotation `e^{jθ̄(s − l_0)}` of data symbol `s` relative to the `l_0`-referenced channel.

**How the timing offset is handled.** It is **measured and reported, not compensated**: nothing in the equalizer reads `TaEst` or `TaPhaseRot`. A timing offset within the cyclic prefix is just a linear phase across frequency, and it is already part of `HEst`, so the equalizer removes it implicitly. `TaEst` goes to the host (`pTaEsts`), and L2 can use it for timing-advance commands.

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

#### How R_ww⁻¹ enters the coefficients

All coefficient kernels compute the same MMSE form:

```
M = Hᴴ R_ww⁻¹,     G = M·H + I,     Ree = G⁻¹,     C = Ree·M
```

No kernel forms R_ww⁻¹ as an explicit matrix. Step 2 estimates R_ww from the DMRS residual (residual → per-PRB covariance + ε·I → optional shrinkage → Cholesky factorization / σ²; see step 2, points 2–6). What it hands over depends on `eqCoeffAlgo` ([channel_eq.cu:554-555](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L554-L555)):

| `eqCoeffAlgo` | R_ww granularity | R_ww⁻¹ used | Comes from |
|---|---|---|---|
| 0 RZF | none | `INV_ZF_REGULARIZER` · I = 3981.07 · I (a fixed σ² ≈ 2.5·10⁻⁴, 36 dB) | constant, [channel_eq_types.cuh:37](../cuPHY/src/cuphy/channel_eq/channel_eq_types.cuh#L37) |
| 1 MMSE, diagonal noise | one scalar per UE group | (1/σ²_w) · I | `invNoiseVarLin` = 1/σ²_w (step 2, point 6) |
| 2 MMSE-IRC | full nRxAnt × nRxAnt matrix per PRB | L_p⁻ᴴ L_p⁻¹, applied by whitening | `LwInv[:, :, p]` = L_p⁻¹ (step 2, point 5) |
| 3 / 4 MMSE-IRC with RBLW / OAS shrinkage | full matrix per PRB, shrunk toward (tr R / A)·I | same as 2 | same as 2, after step 2 point 4 |

**Algorithms 0 and 1 (scalar × identity).** M is just a scaled conjugate transpose ([:744](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L744), [:766](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L766)):

```cpp
shM(r, c) = cuConj(shH(c, r)) * invDiagReg;   // M = Hᴴ / σ²      (invDiagReg = 1/σ²_w or 3981.07)
G = M·H;  G(ℓ,ℓ) += ES_INV;                   // G = HᴴH / σ² + I
```

**Algorithms 2–4 (whitening).** Since R_ww[p] = L_p·L_pᴴ, its inverse is R_ww⁻¹ = (L_p L_pᴴ)⁻¹ = L_p⁻ᴴ L_p⁻¹. The kernel loads the lower-triangular L_p⁻¹ once per PRB (shared by all 12 subcarriers) and whitens the channel ([:3065-3126](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3065-L3126)):

```
N = L_p⁻¹ · H                       whitened channel; lower-triangular left multiply   (:3076)
G = Nᴴ·N + I  = Hᴴ R_ww⁻¹ H + I                                                       (:3097)
M = Nᴴ · L_p⁻¹ = Hᴴ R_ww⁻¹                                                             (:3123)
```

The remaining steps are the same for every algorithm: the LDL solve gives Ree = G⁻¹ and C = Ree·M, followed by the bias correction below.

Using L⁻¹ instead of R_ww⁻¹ has three advantages. The Cholesky factor is computed once per PRB in step 2, not once per subcarrier. A triangular multiply costs about half as much as a full one. Hermitian symmetry is preserved, so G = NᴴN is positive definite by construction.

**Which kernels actually use `LwInv`.** Only one: `eqMmseIrcCoefCompLowMimoKernel` ([:2823-3262](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L2823-L3262)). It is the only place in `channel_eq.cu` that reads `tInfoLwInv`. `eqMmseCoefCompLowMimoKernel` (nRxAnt ≤ 8) dispatches to it when `eqCoeffAlgo` is 2, 3 or 4. For any other algorithm it dispatches to `eqLegacyMmseCoefCompLowMimoKernel` ([:3338-3358](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3338-L3358)).

The 16- and 64-antenna kernels (`eqMmseCoefCompHighMimoKernel*`, `eqMmseCoefCompMassiveMimoKernel_v1`) contain only the scalar `invDiagReg` path ([:940](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L940), [:1360](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L1360), [:1813](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L1813), [:2250](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L2250)). Its flag `noiseRegFlag` is true only for algorithm 1. So with algorithm 2, 3 or 4 these kernels apparently fall back to the RZF constant 3981.07·I, and they do not do IRC. This conclusion comes from reading the kernel code only; it has not been checked against the host-side configuration rules.

#### How Ree and C are solved: LDL factorization of G (`computeLDL`)

Code: [channel_eq.cu:2731-2806](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L2731-L2806), called at [:3136](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3136). The method is described in the comment at [:2912-2926](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L2912-L2926). The description below follows `eqMmseIrcCoefCompLowMimoKernel`. The legacy and high-MIMO kernels build the same augmented matrix ("A = [G | I | M] → [U | Linv | F] → [U | Ree | C]", [:592](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L592)), but their factorization code was not read here.

**Factorization G = Uᴴ·D·U.** U is unit upper triangular (U_ii = 1) and D is real diagonal. Both are stored in place in `shDU` (G's buffer): D_i on the diagonal, U_ij above it. The values 1/D_i are kept in `shDInv`. For i = 0 … L−1:

```
D_i  = G_ii − Σ_{k<i} D_k · |U_ki|²                                     (:2776-2781)
U_ij = ( G_ij − Σ_{k<i} conj(U_ki) · D_k · U_kj ) · (1/D_i),   j > i    (:2792-2799)
```

Parallelism: one warp per subcarrier. For each row i, lane 0 computes D_i, then L − i − 1 lanes compute the rest of row i of U. A `__syncwarp()` separates the rows.

**One solve gives both Ree and C.** The two linear systems share the same matrix, so they are stacked:

```
G·Ree = I,   G·C = M     ⇒     Uᴴ·D·U · [ Ree | C ] = [ I | M ]          (L × (L + nRxAnt) right-hand side)
```

Each lane takes one column of the augmented matrix A = [I | M] and solves it in registers ([:3150-3200](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3150-L3200)):

```
1. Forward substitution   Uᴴ·Y = A     (unit lower triangular, no division):
       Y_i = A_i − Σ_{j<i} conj(U_ji) · Y_j

2. Backward substitution  D·U·X = Y:
       X_i = Y_i · (1/D_i) − Σ_{j>i} U_ij · X_j
```

The columns that came from I give Ree, but only the diagonal Re(X_ii) is kept, in `shMemBlkReeDiag`. The columns that came from M give C. The bias correction λ (next subsection) is then applied to C.

**Why there is no pivoting or guard.** G = NᴴN + I ⪰ I, so every LDL pivot satisfies D_i ≥ 1. The division by D_i is always safe, and Ree_ii ≤ 1 follows. Compare the Cholesky factorization of R_ww in step 2, point 5, which has no such guarantee.

##### Is the full inverse Ree = G⁻¹ required?

**No.** Only two quantities are used afterward. The off-diagonal entries of Ree are never used.

| Needed | Used for |
|---|---|
| C = G⁻¹·M | the equalizer coefficients |
| diag(Ree) only | λ = 1/(1 − Ree_ℓℓ) (bias correction) and `ReeDiagInv` = (1 − Ree)/Ree (LLR scaling) |

**What the code does.** It solves the whole stacked system Uᴴ·D·U·[Ree | C] = [I | M], including all L identity columns, so it does compute the full Ree. It then keeps only the diagonal and discards the off-diagonal results ([:3196-3199](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3196-L3199)):

```cpp
if (col == i) shMemBlkReeDiag(i, FREQ_IDX) = cuReal(regA[i]);
```

**Why that costs little here.** Each column of [I | M] is solved by its own lane. A whole warp handles one subcarrier and has more lanes than columns. For example, 4 layers and 8 antennas give 4 + 8 = 12 columns for 32 lanes. The extra L columns run on lanes that would otherwise be idle, so they add almost no latency. The cost would matter only when columns outnumber lanes, such as with many antennas.

**Alternative 1: diag(Ree) from C·H.** The bias-correction subsection shows that C·H = I − Ree. This also holds in the IRC kernel, because C·H = Ree·Hᴴ R_ww⁻¹ H = I − Ree there too. So

```
Ree_ℓℓ = 1 − Re( Σ_r C_ℓr · H_rℓ )
```

The kernel could solve only G·C = M (nRxAnt right-hand-side columns, no identity columns) and then take one dot product per layer. The cost is FP32 precision. At high SINR, Ree_ℓℓ is small and 1 − (C·H)_ℓℓ loses digits to cancellation: at Ree = 10⁻⁴ (40 dB), the relative error is about 10⁻³. That is still acceptable up to the `MAX_ERROR_PRECISION` cap. The augmented solve avoids the cancellation and is effectively free on idle lanes, which is probably why it was chosen.

**Alternative 2: diag(Ree) from the factorization.** Since G⁻¹ = U⁻¹·D⁻¹·U⁻ᴴ,

```
Ree_ℓℓ = Σ_k |(U⁻¹)_ℓk|² / D_k
```

This needs the triangular inverse U⁻¹, which costs about the same as solving the identity columns.

**Summary.** Only diag(Ree) and C are required. The code gets the full Ree as a side effect of solving both systems at once. That costs almost nothing in the low-MIMO kernel, and everything off the diagonal is discarded.

#### MMSE bias correction: dividing by the MMSE gain

Plain MMSE output is biased: it shrinks each layer's constellation toward zero. Step 5 removes the bias by dividing each layer's coefficients by its **MMSE gain**. That gain is the diagonal element of the *effective channel* C·H, not of the equalizer matrix C. The code never forms C·H. It gets the same value from the diagonal of Ree.

**Definitions in the code** ([channel_eq.cu:568-588](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L568-L588)), with unit-power symbols (R_xx = I):

```
G   = Hᴴ R_ww⁻¹ H + I          (enhanced Gram matrix)
Ree = G⁻¹                      (residual-error covariance)
C   = Ree · Hᴴ R_ww⁻¹          (MMSE coefficient matrix)
```

**The effective channel after equalization:**

```
C·H = Ree · Hᴴ R_ww⁻¹ H = Ree · (G − I) = I − Ree
```

so the equalized output for layer ℓ is

```
z_ℓ = (C·H)_ℓℓ · x_ℓ + inter-layer interference + noise
    = (1 − Ree_ℓℓ) · x_ℓ + …
```

The MMSE gain is g_ℓ = (C·H)_ℓℓ = 1 − Ree_ℓℓ, which is less than 1. This is the bias.

**Correction** ([:3222-3234](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3222-L3234)). Each row ℓ of C is scaled by λ_ℓ = 1/g_ℓ before it is written to `EqCoef`:

```cpp
lambda = one / (one - ree);                // λ = 1 / (1 − Ree_ℓℓ) = 1 / g_ℓ
lambda = min(lambda, MAX_BIAS_LIMIT);      // MAX_BIAS_LIMIT = 10  (channel_eq.cu:67)
Cres   = shMemBlkC(r, c, FREQ_IDX) * lambda;
```

```
C_ub = diag(λ) · C    ⇒    (C_ub·H)_ℓℓ = 1    ⇒    z_ℓ = x_ℓ + e_ℓ   (unbiased)
```

The correction is folded into the stored coefficients, so the soft-demap kernel (step 6) applies none of its own.

**Effect on the error precision** ([:3241-3261](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3241-L3261)). After scaling, the signal power is 1 and the residual-error variance is Ree_ℓℓ/(1 − Ree_ℓℓ)². The stored precision is therefore

```
ReeDiagInv = 1 / (λ · Ree) = (1 − Ree) / Ree = SINR_MMSE − 1     (unbiased post-equalizer SINR)
             capped at MAX_ERROR_PRECISION = 10000 (40 dB)           (channel_eq.cu:66)
```

**The two limits:**

| Constant | Value | Caps | Active when |
|---|---|---|---|
| `MAX_BIAS_LIMIT` | 10 | λ, the bias-correction factor | Ree_ℓℓ ≥ 0.9 (very low SINR) |
| `MAX_ERROR_PRECISION` | 10000 (40 dB) | `ReeDiagInv` | very high SINR; stops oversized LLRs and FP16 overflow (max ≈ 65504) |

The correction is exact only while λ < 10, that is, Ree_ℓℓ < 0.9. Above that, λ stays at 10 and the output remains slightly biased: its gain is 10·(1 − Ree_ℓℓ) < 1. This happens only at very low SINR, where the LLRs are small anyway. The same λ clamp and `ReeDiagInv` cap appear in every coefficient-kernel variant in [channel_eq.cu](../cuPHY/src/cuphy/channel_eq/channel_eq.cu) (around lines 840, 1258, 1673, 2147, 2678 and 3225).

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
| **In** | `CfoEst` | C32F | `[14, nUes]`, indexed via `ueGrpLayerToUeIdx`; multiplies each equalized symbol when CFO correction is enabled (see step 4) |
| **In** | QAM order per layer (`qam[]`) | — | from UE-group parameters |
| **Out** | `LLR` (`tInfoLLR`) | R16F (R32F if built without `LLR_FP16`) | `[8, nLayers, NF, Nd]`; first dimension sized for 256QAM (8 bits per symbol) |
| **Out** | `LLRCdm1` (`tInfoLLRCdm1`) | R16F | `[8, nLayers, NF, Nd]`; only when `nDmrsCdmGrpsNoData` = 1 |
| **Out** | `DataEq` (`tInfoDataEq`) | C16F | `[nLayers, NF, Nd]`; only with debug output, equalize-only mode, or split 7.2e |
| **Out** | `DataEqDft` (`tInfoDataEqDft`) | C32F | `[NF × Nd]`; equalized symbols for DFT-s-OFDM UEs (one UE and one layer per UE group), input to step 7 |

#### What equalization and soft demapping calculate

Code: the `__global__` wrapper `eqMmseSoftDemapKernel` ([channel_eq.cu:5044-5134](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L5044-L5134)) calls the device function `eqMmseSoftDemapKernel_v4` ([:3583-3952](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3583-L3952)). The LLR math is in `soft_demapper_simplified` ([soft_demapper.cuh:1341-1372](../cuPHY/src/cuphy/soft_demapper/soft_demapper.cuh#L1341-L1372)). The build selects it with `EQ_SOFT_DEMAP_USE_TEX = 2` ([channel_eq.cu:65](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L65)). The texture-lookup demapper (`= 1`) is compiled out.

One thread handles one resource element (subcarrier k, data symbol s) of one layer ℓ. It does four things:
1. applies the equalizer;
2. rotates the result to undo CFO;
3. turns the error precision into a PAM noise scale;
4. writes up to 8 LLRs.

**Notation.** Y[r, k, s] = `DataRx` at antenna r, cell subcarrier 12·`startPrb` + k and OFDM symbol s. C[r, k, ℓ, t] = `EqCoef`, the row of the equalizer matrix for layer ℓ, already conjugated, so the equalizer is a plain multiply-accumulate. ρ[k, ℓ, t] = `ReeDiagInv`. l_d = first symbol of DMRS position d.

**1 · Equalizer coefficients for this symbol** (time instance t):

- Without TDI, the single coefficient set t = 0 is used for every data symbol.
- With TDI (`enablePuschTdi` = 1 and at least 2 DMRS positions), the coefficients are interpolated linearly between the two DMRS positions that bracket s ([:3715-3726, 3768](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3715-L3768)):

```
d  = 1, then advance while d < nDmrsPos − 1 and the last symbol of DMRS position d+1 < s
α₁ = (l_d − s) / (l_d − l_{d−1}),      α₂ = (s − l_{d−1}) / (l_d − l_{d−1})
C(s) = α₁ · C[·, d−1] + α₂ · C[·, d]
```

Symbols before l₀ or after the last DMRS use the same formula, so α goes outside [0, 1] there: this is linear extrapolation. In shared memory the coefficients are held as `sC[ant][dmrsIdx][12+1]`. The padding column avoids bank conflicts.

**2 · Equalization (inner product over receive antennas)** ([:3776, 3821](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3776-L3821)):

```
z[ℓ, k, s] = Σ_{r=0}^{nRxAnt−1}  C[r, k, ℓ](s) · Y[r, k, s]
```

Step 5 already folded the MMSE bias correction into C ([:3222-3234](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3222-L3234)). Each coefficient is scaled by λ = 1/(1 − Ree), limited to `MAX_BIAS_LIMIT`. As a result z = x + e is an unbiased estimate of the transmitted QAM symbol x, which has unit average energy.

**3 · CFO de-rotation** (only when `enableCfoCorrection` is set and this is a full-slot pass; [:3851-3855](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3851-L3855)):

```
z ← z · CfoEst[s, u(ℓ)]          (u(ℓ) = ueGrpLayerToUeIdx[ℓ]; see step 4)
```

**4 · Noise scale for the LLRs.**

Step 5 writes the unbiased post-equalizer SINR of each resource element ([:3241-3261](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3241-L3261)):

```
ρ = ReeDiagInv = 1/(λ·Ree) = (1 − Ree)/Ree,   capped at MAX_ERROR_PRECISION = 10000
```

- With TDI, symbols after the first DMRS read ρ[·, d], the same bracket used for the coefficients. All other symbols read ρ[·, 0] ([:3893-3898](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3893-L3898)).
- The complex error variance is 1/ρ. Each real dimension (I or Q, a PAM symbol) carries half of it:

```
σ²_PAM = 1 / (2ρ)        →   PAMnoiseVarInv = 2ρ
c      = 2A · PAMnoiseVarInv = 4A·ρ
```

The factor 2 is applied at [:3482-3488](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3482-L3488). A is the constellation's normalization amplitude. The PAM levels are A·{±1, ±3, …}.

**5 · Soft demapping (simplified max-log LLRs).**

Square QAM with Gray mapping separates into two independent PAMs: I = Re z carries the even bits b₀, b₂, …, and Q = Im z carries the odd bits b₁, b₃, …. This matches the bit order of TS 38.211 §5.1. For each PAM the code uses the piecewise-linear max-log approximation, where a positive LLR means bit = 0 ([soft_demapper.cuh:823-1188](../cuPHY/src/cuphy/soft_demapper/soft_demapper.cuh#L823-L1188)). For x ∈ {I, Q}:

| Modulation (`qam[ℓ]`) | A | LLRs (I gives the even bit, Q the odd bit) |
|---|---|---|
| π/2-BPSK (`nPamBits` = 0) | 1/√2 | b₀ = c·(I + Q) |
| QPSK | 1/√2 | b₀,b₁ = c·x |
| 16QAM | 1/√10 | b₀,b₁ = c·x ; b₂,b₃ = c·(2A − \|x\|) |
| 64QAM | 1/√42 | b₀,b₁ = c·x ; b₂,b₃ = c·(4A − \|x\|) ; b₄,b₅ = c·(2A − \|4A − \|x\|\|) |
| 256QAM | 1/√170 | b₀,b₁ = c·x ; with t₁ = 8A − \|x\| and t₂ = 4A − \|t₁\| : b₂,b₃ = c·t₁ ; b₄,b₅ = c·t₂ ; b₆,b₇ = c·(2A − \|t₂\|) |

- **How the formulas arise.** Each term is the distance from x to the nearest decision boundary of that bit, multiplied by 2A/σ²_PAM. For QPSK this is the exact LLR. For higher orders it is the usual simplified approximation: the MSB term c·x is kept linear even when |x| is beyond the inner points.
- **Precision.** The arithmetic is done in FP16 (`__half2`, I and Q in one instruction).
- **Output layout.** All 8 output slots are always written, as one 16-byte vector. Slots above `qam` bits are 0 ([channel_eq.cu:3471, 3502](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3471-L3502)).
- **Example.** 16QAM with ρ = 100 (20 dB) and z = 0.30 − 0.95j gives c = 4·0.3162·100 = 126.5. The LLRs are: b₀ = +37.9; b₁ = −120.2; b₂ = 126.5·(0.632 − 0.30) = +42.0; b₃ = 126.5·(0.632 − 0.95) = −40.2. That is bits 0, 1, 0, 1, the Gray label of the point (+A, −3A).

**6 · Where the LLRs are written.**

- **`LLR`:** `LLR[0..7, ℓ, k, i]`, where i = position of s in the UE group's `dataSymLoc` list ([:3860-3863](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3860-L3863)).
- **`nDmrsCdmGrpsNoData` = 1:** data also occupies the DMRS symbols, on the odd subcarriers that CDM group 0 leaves free.
  - The kernel extends its symbol count to `nDataSym + nDmrsSyms` and still writes the full grid to `LLR`.
  - It also writes a **packed** copy to `LLRCdm1` ([:3867-3890](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3867-L3890)). On DMRS symbols this copy keeps only odd k, stored at k/2, so a DMRS symbol takes half the stride of a data symbol. The per-symbol start offsets are a prefix sum built in shared memory (`smem_llr_addr_offset`, [:3641-3679](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3641-L3679)).
  - The packed copy follows the 38.211 RE order (DMRS REs skipped), which is presumably why it exists for the downstream de-rate-matching. That consumer was not checked here.
- **DFT-s-OFDM UEs** (`enableTfPrcd` = 1): no LLR is computed here. z is stored to `DataEqDft[k + i·12·nPrb]` ([:3939-3943](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3939-L3943)). The step 7 kernels then run the IDFT and demap the result with the same LLR formulas.
- **Debug output:** with `enableDebugEqOutput`, z is also written to `DataEq[ℓ, k, i]`.

#### Tensors and dimensions (default kernel)

| | Tensor | Type | Shape | Indexed by this kernel as |
|---|---|---|---|---|
| In | `DataRx` | C16F | `[NF_cell = 12·nPrbCell, 14, nRxAnt]` | `(12·startPrb + 12·prb + k, s, r)`, cell-absolute subcarrier |
| In | `EqCoef` | C32F | `[nRxAnt, 12, nLayers, nPrb, NH]` | `(r, k, ℓ, prb, t)`; t = 0, or the TDI pair `d−1, d` |
| In | `ReeDiagInv` | R32F | `[12, nLayers, nPrb, nTimeChEq]` | `(k, ℓ, prb, 0 or d)` |
| In | `CfoEst` | C32F | `[14, nUes]` | `(s, ueGrpLayerToUeIdx[ℓ])`, full slot only |
| In | UE-group parameters | — | `qam[ℓ]`, `dataSymLoc[]`, `dmrsSymLoc[]`, `startPrb`, `nPrb`, `nDataSym`, `nDmrsCdmGrpsNoData`, `enableTfPrcd`, `enablePuschTdi`, `enableCfoCorrection` | from `cuphyPuschRxUeGrpPrms_t` |
| Out | `LLR` | R16F | `[8, nLayers, 12·nPrb, Nd]` | `(bit, ℓ, 12·prb + k, i)`, UE-group-relative subcarrier |
| Out | `LLRCdm1` | R16F | same allocation, packed as described above | only when `nDmrsCdmGrpsNoData` = 1 |
| Out | `DataEqDft` | C32F | `[12·nPrb · Nd]` | DFT-s-OFDM UE groups only |
| Out | `DataEq` | C16F | `[nLayers, 12·nPrb, Nd]` | debug only |

In this table, Nd = `nDataSym`, plus `nDmrsSyms` when `nDmrsCdmGrpsNoData` = 1.

#### How many kernels are launched

**Launch geometry** (`softDemapKernelLaunchGeo`, [channel_eq.cu:5431-5459](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L5431-L5459)):

```
grid  = (nMaxPrb, nMaxLayers, nUeGrpsInCfg)    blockIdx.x = PRB, blockIdx.y = layer, blockIdx.z → hetCfgUeGrpMap → UE group
block = (12, nMaxDataSym)                       threadIdx.x = subcarrier in PRB, threadIdx.y = data-symbol index i
```

- At most 12·14 = 168 threads per block (`__launch_bounds__`).
- The maxima are taken over the UE groups that share the launch ([:7852-7854](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L7852-L7854)). A block whose PRB or layer does not exist in its UE group exits immediately, and so does a thread whose i ≥ that group's `nDataSym` ([:3592-3598, 3687](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3592-L3687)).
- The 7.2e variant uses block `(12, layersPerCta, Nd)` and grid `(nPrb, layerGroups, nUeGrps)` instead.

**Grouping into launches.** UE groups are grouped by **kernel function pointer** (`findKernelFunc`, [:7838-7880](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L7838-L7880)), one launch per group of identical pointers ("heterogeneous config"). The pointer is set by template parameters only:

| Splits launches | Does **not** split launches (runtime branches inside one kernel) |
|---|---|
| `nRxAnt` (template `N_BS_ANTS`: 1–8, 16, or 64 → `eqMmseSoftDemapKernel_64R`) | number of PRBs, `startPrb`, layer count, `nDataSym` |
| Symbol bitmask (full slot `0x3FFF` vs early HARQ `0xF`/`0x1F`), set per pass, not per UE group | QAM order per layer |
| `enableDebugEqOutput`, `kernelSelOption` (equalize-only), 7.2e split, data types; all global | TDI on/off, CFO correction on/off, DFT-s-OFDM on/off, `nDmrsCdmGrpsNoData` |

So for one full-slot pass:

```
#launches(step 6) = number of distinct nRxAnt values among the UE groups      (usually 1)
```

- **Early HARQ** adds the same number of launches again in the sub-slot graph, with bitmask `0xF`/`0x1F`. That variant has TDI and CFO correction fixed off ([:5071-5081](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L5071-L5081)).
- **DFT-s-OFDM UE groups** share the step 6 launch, which writes their `DataEqDft`. Step 7 then adds its own `bluestein_Idft_kernel` and `eqMmseSoftDemapAfterDftKernel` launches.

**Example.** Three UE groups on a 4-antenna cell:

| UE group | PRBs | Layers | Modulation | Other |
|---|---|---|---|---|
| A | 40 | 1 | QPSK | |
| B | 60 | 2 | 64QAM | with TDI |
| C | 100 | 4 | 256QAM | with `nDmrsCdmGrpsNoData` = 1, so Nd = 12 + 1 |

- **Launches:** 1, because all three groups use `eqMmseSoftDemapKernel<…, 4, 0x3FFF, false>`.
- **Geometry:** grid = (100, 4, 3) and block = (12, 13).
- **Early exits:** group A uses 40 × 1 of its 400 blocks, and in group A and group B the threads with i = 12 exit at once.

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
