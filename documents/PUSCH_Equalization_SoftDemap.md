# PUSCH Equalization and Soft Demapping: From DataRx and EqCoef to LLRs

This document explains how cuPHY's PUSCH equalizer (`cuPHY/src/cuphy/channel_eq/`) turns the received samples into LLRs. It covers:

- how `EqCoef` relates to `HEst`,
- what each kernel computes,
- how the work is split into thread blocks and threads (per PRB, per subcarrier, per data symbol, per layer),
- a worked example with numbers.

It complements [PUSCH_ChEst_ChEq_workflow.md](PUSCH_ChEst_ChEq_workflow.md), which covers the whole pipeline. Source references are to [channel_eq.cu](../cuPHY/src/cuphy/channel_eq/channel_eq.cu) and [channel_eq.hpp](../cuPHY/src/cuphy/channel_eq/channel_eq.hpp) unless noted. Tensor shapes list the fastest-changing dimension first.

## 1. Overview

Equalization runs as two kernels, launched in this order for each slot:

```
HEst ──► (A) Coefficient computation ──► EqCoef, ReeDiagInv
                                              │
DataRx ─────────────────────────────────────► (B) Equalize + soft demap ──► LLR
CfoEst ─────────────────────────────────────►
```

| Kernel | Purpose | Partitioned over |
|---|---|---|
| **(A)** `eqMmseCoefCompLowMimoKernel` (or the High/Massive-MIMO variants) | Compute the MMSE receive filter for every subcarrier | PRBs × subcarriers (no symbol dimension) |
| **(B)** `eqMmseSoftDemapKernel` (or `_64R`, `72e`, equalize-only variants) | Apply the filter to every received data RE and convert the result to LLRs | PRBs × layers × subcarriers × data symbols |

## 2. HEst vs. EqCoef

They describe the same channel from opposite directions:

| | `HEst` | `EqCoef` |
|---|---|---|
| Meaning | The **channel**: how each transmitted layer arrives at each receive antenna | The **receive filter**: how to combine the antennas to recover each layer |
| Per subcarrier | matrix **H**, `nRxAnt` × `nLayers` | matrix **C**, `nLayers` × `nRxAnt` |
| Model | **y** = **H**·**x** + **n** | **x̂** = **C**·**y** |
| Produced by | channel estimation (`ch_est/`) from the DMRS | kernel (A), from `HEst` and the noise estimate |
| Shape | `[nRxAnt, nLayers, NF, NH]` | `[nRxAnt, 12, nLayers, nPrb, NH]` |

The two tensors have the same number of elements; `EqCoef` just splits the subcarrier dimension `NF` into (12 subcarriers, `nPrb` PRBs) and puts antennas first. Element `EqCoef[ant, k, layer, prb, h]` is entry (layer, ant) of **C** on subcarrier 12·prb + k.

**C** is not simply the inverse of **H**. It is the MMSE solution, which balances undoing the channel against amplifying noise:

```
C = (Hᴴ · Rww⁻¹ · H + I)⁻¹ · Hᴴ · Rww⁻¹          (then scaled per layer to remove bias, see 3.2)
```

`Rww` is the noise-plus-interference covariance at the receive antennas. With no noise this reduces to the zero-forcing pseudo-inverse of **H**.

**NH** (last dimension of both) is the number of DMRS positions, `dmrsAddlnPos` + 1. One coefficient set exists per DMRS position, not per OFDM symbol: data symbols get theirs in kernel (B), by reuse or interpolation (section 4.2). Without time-domain interpolation (`enablePuschTdi` = 0, the default), only set 0 is computed.

## 3. Kernel (A): coefficient computation

### 3.1 Partition into blocks and threads

From `coefCompLowMimoKernelLaunchGeo()` and `getThreadsPerFreqBin()`:

```
gridDim  = (nPrb, nUeGrps)                                   one thread block per PRB per UE group
blockDim = (threadsPerSubcarrier, subcarriersPerIteration)
```

| Parameter | Value |
|---|---|
| `threadsPerSubcarrier` (`getThreadsPerFreqBin`) | `max(nRxAnt·nLayers, nRxAnt + nLayers)` rounded up to a power of two, at most 32, so one subcarrier's matrix work stays within one warp |
| `subcarriersPerIteration` (`N_FREQ_BINS_PER_ITER`) | 12 if `nRxAnt` ≤ 4, else 4 |
| Iterations per block | 12 / `subcarriersPerIteration` (1 or 3), so every block covers all 12 subcarriers of its PRB |

So the work is split **by PRB and by subcarrier**: each subcarrier is an independent small matrix problem solved by a group of `threadsPerSubcarrier` threads. There is **no symbol dimension**: the result depends only on the channel, which is estimated per DMRS position, not per symbol. The kernel is launched once (TDI off) or once per DMRS position (TDI on, `chEqTimeInstIdx` = 0 … `NH`−1).

The 16-antenna and 64-antenna variants (`eqMmseCoefCompHighMimoKernel`, `eqMmseCoefCompMassiveMimoKernel_v1`) compute the same quantities with different thread mappings suited to larger matrices.

### 3.2 What each subcarrier's thread group computes

Code: `eqLegacyMmseCoefCompLowMimoKernel` (RZF and MMSE) and `eqMmseIrcCoefCompLowMimoKernel` (MMSE-IRC).

1. **Load H.** The block loads the `nRxAnt` × `nLayers` matrices of its 12 subcarriers from `HEst[:, :, k, h]` into shared memory. For time instance `h` > 0 with CFO correction enabled, each layer's column is multiplied by `CfoEst` at that DMRS symbol, so all coefficient sets share a common phase reference.

2. **Form the Gram matrix.**

   | `eqCoeffAlgo` | Noise model `Rww⁻¹` | Computation |
   |---|---|---|
   | 0 RZF | constant `3981·I` (`INV_ZF_REGULARIZER`, about −36 dB noise) | M = Hᴴ·Rww⁻¹, G = M·H + I |
   | 1 MMSE, diagonal noise | `(1/σ²)·I`, σ² from the noise estimate (`invNoiseVarLin`) | same |
   | 2–4 MMSE-IRC | full per-PRB covariance via its inverse Cholesky factor `LwInv` [`nRxAnt`, `nRxAnt`, `nPrb`] | N = LwInv·H, M = Nᴴ·LwInv, G = Nᴴ·N + I |

   The `+ I` term is `Rxx⁻¹`, the inverse of the transmitted-symbol covariance (unit energy). MMSE-IRC whitens the interference with `LwInv`, so it can suppress interference that is correlated across antennas.

3. **Solve for Ree and C together.** Factorize G = Uᴴ·D·U (LDLᴴ), then solve the augmented system

   ```
   Uᴴ·D·U · [Ree | C] = [I | M]
   ```

   by forward substitution, then backward substitution. This yields **Ree** = G⁻¹ (the error covariance after equalization) and **C** = Ree·M without forming an explicit inverse.

4. **Bias correction and outputs.** An MMSE estimate is scaled down by (1 − Ree_ll) for layer *l*. To undo that, each layer is scaled by

   ```
   λ_l = 1 / (1 − Ree_ll)            (capped at MAX_BIAS_LIMIT = 10)
   EqCoef[:, k, l, prb, h] = λ_l · C[l, :]
   ReeDiagInv[k, l, prb, h] = 1 / (λ_l · Ree_ll)   (capped at MAX_ERROR_PRECISION = 10000)
   ```

   Since 1/(λ·Ree) = 1/Ree − 1, `ReeDiagInv` is the **post-equalization SINR** of layer *l* on that subcarrier. Kernel (B) uses it as the inverse noise variance when computing LLRs.

**Outputs:** `EqCoef` C32F `[nRxAnt, 12, nLayers, nPrb, NH]` and `ReeDiagInv` R32F `[12, nLayers, nPrb, nTimeChEq]` (`nTimeChEq` = 4 with TDI, else 1).

## 4. Kernel (B): equalization and soft demapping

### 4.1 Partition into blocks and threads

From `softDemapKernelLaunchGeo()`:

```
gridDim  = (nPrb, nLayers, nUeGrps)      one thread block per PRB, per layer, per UE group
blockDim = (12, Nd)                      threadIdx.x = subcarrier in the PRB, threadIdx.y = data symbol
```

So this kernel **is** partitioned by both subcarrier and symbol: **one thread per resource element (subcarrier × data symbol) per layer**.

- `threadIdx.y` runs over the **`Nd` data symbols, not all 14**. It indexes `dataSymLoc[]`, the list of symbols carrying data. DMRS symbols are skipped, except when `nDmrsCdmGrpsNoData` = 1: then data also occupies the unused CDM group of the DMRS symbols, those symbols are included in `Nd`, and their LLRs use half the subcarriers (written via `LLRCdm1` and a per-symbol address offset).
- Each layer gets its own thread blocks (`blockIdx.y`), so a thread produces one layer's estimate for one RE.

### 4.2 What each thread computes

Code: `eqMmseSoftDemapKernel` → `eqMmseSoftDemapKernel_v4`. For thread (subcarrier *k*, data symbol *s*) in the block for (PRB *p*, layer *l*):

1. **Pick the coefficients for this symbol.**
   - **TDI off:** use set 0, `EqCoef[:, k, l, p, 0]`, for every symbol.
   - **TDI on:** the block first loads all `NH` sets for its subcarriers into shared memory, then each thread interpolates linearly between the two DMRS positions *t*₁ < *t*₂ bracketing (or nearest to) its symbol:
     ```
     c(s) = α₁·c[t₁] + α₂·c[t₂],   α₁ = (t₂ − s)/(t₂ − t₁),   α₂ = (s − t₁)/(t₂ − t₁)
     ```
2. **Equalize** with a dot product over the receive antennas:
   ```
   x̂ = Σ_ant  c[ant] · DataRx[k_abs, s, ant]
   ```
   where `k_abs` is the subcarrier's index in the cell's slot buffer (`startPrb`·12 + 12·*p* + *k*).
3. **CFO correction** (if enabled, full-slot path only): `x̂ ← x̂ · CfoEst[s, ue]`, a per-symbol phase rotation for the UE that owns layer *l*.
4. **Noise level:** read `ReeDiagInv[k, l, p, ·]`, from set 0 without TDI, or from the set matching the symbol's DMRS interval with TDI.
5. **Soft demapping** (`ch_eq_simplified_soft_demapper`, [soft_demapper.cuh](../cuPHY/src/cuphy/soft_demapper/soft_demapper.cuh)): the I and Q components are demapped separately as PAM signals, with PAM noise-variance inverse 2·`ReeDiagInv`, using closed-form LLR expressions for each modulation (π/2-BPSK, QPSK, 16/64/256QAM). This produces `Qm` LLRs (`Qm` = 1–8, from `qam[l]`).
6. **Write** a group of 8 LLR slots to `LLR[0..7, l, 12·p + k, s]`, indexed by the subcarrier within the UE group's allocation (no `startPrb` offset). All 8 slots are written; only the first `Qm` are meaningful.

**Output:** `LLR` R16F `[8, nLayers, NF, Nd]`.

Variants with the same structure: `eqMmseSoftDemapKernel_64R` (64 antennas, different thread mapping), `eqMmseEqualizationKernel` (stops after step 3 and writes `DataEq`), `eqMmseSoftDemap72eKernel` (O-RAN split 7.2e), and for DFT-s-OFDM an equalization pass followed by `bluestein_Idft_kernel` and `eqMmseSoftDemapAfterDftKernel`.

## 5. Worked example

One UE group: `nPrb` = 32, `nRxAnt` = 4, `nLayers` = 4, 256QAM, single-symbol DMRS on symbols 2 and 11 (`NH` = 2), `nDmrsCdmGrpsNoData` = 2, TDI off.

**Kernel (A), launched once:**
- `threadsPerSubcarrier` = max(4·4, 4+4) = 16; `subcarriersPerIteration` = 12; 1 iteration.
- grid (32, 1), block (16, 12): 32 blocks × 192 threads.
- 32 PRBs × 12 = **384 independent 4×4 MMSE problems**, one per subcarrier, each solved by 16 threads.
- Output: `EqCoef` [4, 12, 4, 32, 2] with only `h` = 0 filled; `ReeDiagInv` [12, 4, 32, 1].

**Kernel (B):**
- Data symbols: 14 − 2 DMRS = **`Nd` = 12** (symbols 0, 1, 3–10, 12, 13).
- grid (32, 4, 1), block (12, 12): 128 blocks × 144 threads = **18,432 threads** = 384 subcarriers × 12 symbols × 4 layers.
- Each thread: a 4-term complex dot product, an optional CFO phase rotation, then 8 LLRs.
- Output: `LLR` [8, 4, 384, 12].

With TDI on, kernel (A) would also run for `h` = 1, and each thread in kernel (B) would interpolate between the coefficients at symbols 2 and 11 according to its own symbol index.

## 6. Summary

| Question | Answer |
|---|---|
| Is `EqCoef` the same as `HEst`? | No. `HEst` is the channel **H**; `EqCoef` is the MMSE receive filter **C** computed from it (and from the noise estimate), with per-layer bias correction. |
| How is the coefficient kernel partitioned? | One block per PRB (and UE group); within it, one small thread group per subcarrier. No symbol dimension. |
| How is the equalize + soft-demap kernel partitioned? | One block per PRB × layer (× UE group); within it, one thread per subcarrier × data symbol. |
| Does it use all 14 symbols? | No. Only the `Nd` data symbols (`dataSymLoc[]`); DMRS symbols are included only when they also carry data (`nDmrsCdmGrpsNoData` = 1). |
| Where does symbol-to-symbol variation come in? | Not in the coefficients (one set per DMRS position), but in kernel (B): per-symbol CFO phase correction, and optional linear interpolation between DMRS positions (TDI). |
| What is `ReeDiagInv`? | The bias-corrected post-equalization SINR per layer and subcarrier, used as the inverse noise variance for the LLRs. |
