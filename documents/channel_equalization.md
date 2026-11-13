# PUSCH Channel Equalization (cuPHY `channel_eq`)

This document describes the PUSCH receive channel-equalization pipeline implemented in
`cuPHY/src/cuphy/channel_eq/`, from the frequency-domain received symbols and the channel
estimate all the way to the output LLRs. It also explains how the work is partitioned into
CUDA thread blocks and threads, and clarifies the relationship between **HEst** and **EqCoef**.

The pipeline is a **two-stage** process:

```
                Stage 1: Coefficient Compute                 Stage 2: Equalize + Soft Demap
   HEst (H)  ───────────────────────────────►  EqCoef (C)  ───────────────────────────────►  LLR
   (input)        MMSE / MMSE-IRC solve        (EqCoef)        C·Y dot-product + demap
                                    └────►  ReeDiagInv (noise var) ────┘
```

- **Stage 1** turns the channel estimate `HEst` into the equalizer coefficient matrix `EqCoef`
  (plus a per-layer noise-variance term `ReeDiagInv`).
- **Stage 2** consumes `DataRx` (the received frequency-domain symbols) together with `EqCoef`
  and `ReeDiagInv` to produce the per-layer LLRs.

The two stages are separate kernel launches (see `puschRxChEq::setupCoefCompute` and
`puschRxChEq::setupSoftDemap` in `channel_eq.cu`).

---

## 1. Inputs and outputs

All tensors are described by `cuphyTensorInfo2_t` (pointer + strides) inside the derived
per-UE-group parameter struct `cuphyPuschRxUeGrpPrms_t`.

| Tensor | Meaning | Shape (as annotated in code) | Role |
|--------|---------|------------------------------|------|
| `tInfoHEst` | **HEst** – channel estimate matrix `H` | `(N_BS_ANTS, N_LAYERS, NF, NH)` | **Input to Stage 1.** One `N_BS_ANTS × N_LAYERS` complex matrix per subcarrier (frequency bin). |
| `tInfoEqCoef` | **EqCoef** – MMSE equalizer coefficient matrix `C` | `(N_SC, N_LAYERS, N_BS_ANTS, N_PRB, NH)` | **Output of Stage 1, input to Stage 2.** One `N_LAYERS × N_BS_ANTS` complex matrix per subcarrier. |
| `tInfoReeDiagInv` | Inverse diagonal of the residual error covariance `Ree = G⁻¹` | `(N_SC, N_LAYERS, N_PRB, NH)` | **Output of Stage 1, input to Stage 2.** Used as the noise variance to scale the LLRs. |
| `tInfoDataRx` | **DataRx** – received frequency-domain symbols `Y` | `(NF, ND, N_BS_ANTS)` | **Input to Stage 2.** Received samples per subcarrier, per symbol, per RX antenna. |
| `tInfoLLR` | **LLR** – log-likelihood ratios | `(N_LLR, N_LAYERS, NF, ND)` | **Output of Stage 2.** |
| `tInfoLLRCdm1` | LLRs for the second CDM group (when `nDmrsCdmGrpsNoData == 1`) | `(N_LLR, N_LAYERS, NF, ND)` | Optional second LLR output. |

Other inputs used along the way: `tInfoLwInv` (inverse Cholesky factor of the
noise/interference covariance, for MMSE-IRC), `tInfoCfoEst` (CFO correction), and the
DMRS/data symbol location tables (`dmrsSymLoc`, `dataSymLoc`).

### Key constants

- `CUPHY_N_TONES_PER_PRB = 12` — subcarriers per PRB (`cuphy.h:1133`).
- `OFDM_SYMBOLS_PER_SLOT = 14` — OFDM symbols per slot (`cuphy.h:58`).

---

## 2. Stage 1 — Equalizer coefficient computation (HEst → EqCoef)

Entry: `puschRxChEq::setupCoefCompute` → `batchEqCoefComp` → `coefCompKernelSelectL1/L0` →
one of the `eqMmseCoefComp*` kernels.

### 2.1 What it computes

For **each subcarrier independently**, the kernel solves the MMSE (or MMSE-IRC) filter from
the channel matrix `H` for that subcarrier. The per-subcarrier linear algebra is:

- `H` : channel matrix, `N_BS_ANTS × N_LAYERS` (rows = RX antennas, cols = layers).
- `M = H' · Rww⁻¹` : `N_LAYERS × N_BS_ANTS`.
- `G = H' · Rww⁻¹ · H + Rxx⁻¹` : enhanced Gram matrix, `N_LAYERS × N_LAYERS`.
- Joint LU factorization of the augmented matrix `A = [ G | I | M ]` → `[ U | G⁻¹ | C ]`.
- `Ree = G⁻¹` (residual error covariance) and `C = Ree · M` (the MMSE coefficient matrix).

The kernel then writes:
- `EqCoef` = `C` (with an optional bias-correction factor `lambda`), and
- `ReeDiagInv` = the inverse of the (bias-corrected) diagonal of `Ree`, saturated to
  `MAX_ERROR_PRECISION`.

There are three kernel flavors selected by antenna/layer count
(`coefCompKernelSelectL1`, `channel_eq.cu:6863`):

| Flavor | Used for | Notes |
|--------|----------|-------|
| `eqMmseCoefCompLowMimo` | small MIMO, e.g. `{layers,ants}` = {1,2},{2,2},{1,4},{2,4},{4,4},{1,8},{2,8},{4,8} | One block per PRB; supports MMSE-IRC via `tInfoLwInv`. |
| `eqMmseCoefCompHighMimo` | high MIMO, e.g. {8,16},{16,16} | Fewer threads; parallel LU factorization. |
| `eqMmseCoefCompMassiveMimo` | massive MIMO | One tone per block/iteration. |

### 2.2 Block / thread partitioning (Stage 1)

Stage 1 is partitioned **over subcarriers (frequency bins) only** — there is **no symbol
dimension**, because the channel estimate `H` is defined per subcarrier (from the DMRS), not
per data symbol.

**Low MIMO** (`coefCompLowMimoKernelLaunchGeo`, `channel_eq.cu:5374`):

```
gridDim  = ( nPrb, nUeGrps )
blockDim = ( threadsPerFreqBin, N_FREQ_BINS_PER_ITER )
```

- `blockIdx.x` = PRB index, `blockIdx.y` = UE group.
- One thread block processes a **whole PRB (12 subcarriers)** by looping over frequency bins
  in iterations of `N_FREQ_BINS_PER_ITER` (12 if `N_BS_ANTS ≤ 4`, else 4).
- `blockDim.x` = `getThreadsPerFreqBin<N_BS_ANTS,N_LAYERS>()` — a divisor of 32, so each
  frequency-bin group fits inside a single warp (allows `__syncwarp` instead of
  `__syncthreads`). `blockDim.y` = number of bins processed per iteration.

**High MIMO** (`coefCompHighMimoKernelLaunchGeo`, `channel_eq.cu:5302`):

```
gridDim  = ( N_THRD_BLK_PER_PRB, nPrb, nUeGrps )   // N_THRD_BLK_TONES = 6
blockDim = ( N_THRDS_PER_TONE, N_TONES_PER_ITER )   // N_TONES_PER_ITER = 2
```

**Massive MIMO** (`coefCompMassiveMimoKernelLaunchGeo`, `channel_eq.cu:5227`):

```
gridDim  = ( N_THRD_BLK_PER_PRB, nPrb, nUeGrps )   // N_THRD_BLK_TONES = 1
blockDim = ( N_THRDS_PER_TONE, N_TONES_PER_ITER )   // N_TONES_PER_ITER = 1
```

So in Stage 1 the grid is indexed by **PRB / subcarrier** (and UE group); the 14-slot symbol
structure does not appear.

---

## 3. Stage 2 — Equalization + soft demapping (DataRx + EqCoef → LLR)

Entry: `puschRxChEq::setupSoftDemap` → `batchEqSoftDemap` → `eqMmseSoftDemap` →
`eqMmseSoftDemapKernel_v4` (`channel_eq.cu:3572`).

### 3.1 What it computes

For each (subcarrier, data symbol, layer):

1. **Equalization** — form the soft (equalized) estimate as the dot product of the equalizer
   coefficient vector and the received vector across RX antennas:

   ```
   softEst = Σ_ant  C[ant] · Y[ant]      // C = EqCoef, Y = DataRx
   ```

   - In the **TDI** path, the coefficient is first linearly interpolated in time between the
     two surrounding DMRS symbols (`alpha1`/`alpha2`), and the coefficients are staged in
     shared memory (`sC`).
   - Optional **CFO correction**: `softEst *= CfoEst[symIdx, ueIdx]`.

2. **Soft demapping** — convert `softEst` to LLRs using `ReeDiagInv` as the noise variance:

   ```
   PAM inv-variance = 2 · ReeDiagInv      // QAM variance / 2 → PAM variance
   LLR = soft_demapper( softEst, 2·ReeDiagInv, nPamBits )
   ```

   The demapper is selected by `EQ_SOFT_DEMAP_USE_TEX` (currently `2` = simplified soft
   demapper, `ch_eq_simplified_soft_demapper`; `1` = texture-based; `0` = legacy). LLRs are
   written to `tInfoLLR` (and optionally `tInfoLLRCdm1` for the second CDM group).

   `qam[layerIdx]` selects the modulation order (QAM4/16/64/256 → 1/2/3/4 bits per PAM axis).

### 3.2 Block / thread partitioning (Stage 2)

Stage 2 is partitioned over **both subcarriers and symbols** (and layer / UE group). This is
where the **14-slot symbol structure** enters.

**Default (non-7.2e) launch** (`softDemapKernelLaunchGeo`, `channel_eq.cu:5431`):

```
gridDim  = ( nPrb, nLayers, nUeGrps )
blockDim = ( CUPHY_N_TONES_PER_PRB, Nd )     // (12, nDataSym)
```

- `blockIdx.x` = PRB, `blockIdx.y` = layer, `blockIdx.z` = UE group.
- `threadIdx.x` = **subcarrier within the PRB** (0..11) → `FREQ_IDX`.
- `threadIdx.y` = **data-symbol index** (0..Nd-1) → `DATA_SYMB_ABS_IDX`, where `Nd = nDataSym`
  (the number of data symbols, ≤ `OFDM_SYMBOLS_PER_SLOT = 14`).

So **one thread handles exactly one (subcarrier, data symbol) for one layer**, computing that
symbol's `softEst` and LLR. The block covers one PRB × all data symbols of one layer.

**7.2-e functional split** (`openRanFunctionalSplitOption == PUSCH_7_2_E`):

```
gridDim  = ( nPrb, layerGroups, nUeGrps )     // layerGroups = ceil(nLayers / 2)
blockDim = ( CUPHY_N_TONES_PER_PRB, maxLayersPerCta, Nd )   // (12, ≤2, nDataSym)
```

Layers are grouped (up to 2 per CTA) so the z-dimension carries the symbol index.

**DFT-s-OFDM variants** (separate entry points `setupSoftDemapIdft` /
`setupSoftDemapAfterDft`) use a different geometry:
- `softDemapAfterDftKernelLaunchGeo` (`channel_eq.cu:5462`):
  `gridDim = (nPrb, ceil(Nd/N_SYMBS_PER_THRD_BLK), nUeGrps)`,
  `blockDim = (CUPHY_N_TONES_PER_PRB, N_SYMBS_PER_THRD_BLK)` — a block processes one PRB and a
  small batch of symbols.
- `softDemapKernelLaunchGeo_64R` (`channel_eq.cu:5485`): for `N_BS_ANTS = 64`,
  `blockDim.x = (64/2)·12`, with `N_SYMBS_PER_THRD_BLK` symbols per block.

---

## 4. Answer to the specific questions

**Q: Based on `DataRx` and `EqCoef`, how is the processing partitioned into blocks and threads?**

In the Stage-2 equalize+demap kernel (the one that actually consumes `DataRx` and `EqCoef`):

- **Thread block** = one **PRB** × one **layer** (× one UE group). For the 7.2-e split, a block
  covers one PRB × up to 2 layers.
- **Threads within a block** = **subcarrier (x) × data symbol (y)**:
  - `threadIdx.x` → subcarrier index within the PRB (12 values).
  - `threadIdx.y` → data-symbol index (up to 14 values).
- Each thread reads the `EqCoef` column and `DataRx` column for its (subcarrier, symbol), forms
  `softEst = C·Y`, and writes the LLR.

**Q: Is it partitioned based on 14 symbols and/or subcarriers?**

Both, but in different stages:

- **Subcarriers** are the primary partitioning axis in **both** stages (Stage 1 by PRB/frequency
  bin; Stage 2 by `threadIdx.x`).
- **Symbols** (the 14-slot structure, `OFDM_SYMBOLS_PER_SLOT = 14`) appear **only in Stage 2**,
  as the `threadIdx.y` (data-symbol) dimension. Stage 1 has no symbol axis because the channel
  estimate is per-subcarrier.

**Q: Is `EqCoef` the HEst? What is their relationship?**

No — they are different tensors:

- **HEst** (`tInfoHEst`) is the **channel estimate** `H`, an `N_BS_ANTS × N_LAYERS` complex
  matrix per subcarrier. It is the **input** to Stage 1.
- **EqCoef** (`tInfoEqCoef`) is the **MMSE equalizer coefficient** `C`, an `N_LAYERS × N_BS_ANTS`
  complex matrix per subcarrier. It is the **output** of Stage 1 and the **input** to Stage 2.
- Relationship: `EqCoef = C = Ree · M`, where `M = H'·Rww⁻¹` and `Ree = (H'·Rww⁻¹·H + Rxx⁻¹)⁻¹`.
  In other words, **EqCoef is the MMSE (or MMSE-IRC) filter derived from HEst**. Stage 1
  performs this derivation per subcarrier; Stage 2 then applies `EqCoef` to `DataRx`.

---

## 5. File / function map

| File | Contents |
|------|----------|
| `channel_eq.hpp` | Public `puschRxChEq` class, descriptors, kernel-arg structs, `getThreadsPerFreqBin`. |
| `channel_eq_types.cuh` | Constants, QAM tag dispatch, debug helpers. |
| `channel_eq.cu` | All kernels: LU factorization, MMSE coefficient compute (low/high/massive MIMO), soft-demapping kernels, Bluestein DFT-s-OFDM IDFT, and the `setup*`/`batch*`/`*KernelLaunchGeo` launch logic. |
| `dft_s_ofdm_bluestein_workspace.hpp` | Bluestein workspace for DFT-s-OFDM IDFT. |

Key functions:
- `puschRxChEq::setupCoefCompute` (`channel_eq.cu:7674`) — Stage 1 setup.
- `puschRxChEq::coefCompKernelSelectL1` (`channel_eq.cu:6863`) — picks low/high/massive MIMO kernel.
- `eqMmseCoefCompLowMimoKernel` (`channel_eq.cu:3328`) — Stage 1 kernel.
- `puschRxChEq::setupSoftDemap` (`channel_eq.cu:8328`) — Stage 2 setup.
- `eqMmseSoftDemapKernel_v4` (`channel_eq.cu:3572`) — Stage 2 equalize+demap kernel.
- `softDemapKernelLaunchGeo` (`channel_eq.cu:5431`) — Stage 2 grid/block geometry.
