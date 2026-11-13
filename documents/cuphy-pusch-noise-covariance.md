# Noise variance and interference-plus-noise covariance for the cuPHY PUSCH LMMSE equalizer

How the cuPHY PUSCH pipeline obtains the noise variance (σ²) and the
interference-plus-noise covariance (R_ww) that its LMMSE equalizer uses, read
from the source of `aerial-cuda-accelerated-ran`, Release 26-2 (commit
`4f65f97`). Paths are relative to `cuPHY/src/`; line numbers are for that
commit. Statements below are what the code does, not what the 3GPP spec or the
Aerial documentation says.

## Summary

The equalizer does **not** estimate noise itself. A separate kernel,
`pusch_noise_intf_est`, runs **after channel estimation and before the
equalizer**. It forms the DMRS residual with the *same slot's* channel estimate
and turns it into either a scalar σ² or a per-PRB R_ww. The equalizer algorithm
(`eqCoeffAlgo`) decides which of the two is used.

| `eqCoeffAlgo` | equalizer uses | how it is obtained |
|---|---|---|
| `0` `PUSCH_EQ_ALGO_TYPE_RZF` | a **fixed** regulariser, σ² = 1/3981 (−36 dB) | constant `INV_ZF_REGULARIZER`, nothing estimated |
| `1` `PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE` | **scalar** σ², `R_ww = σ² I` | mean of diag(R_ww) over Rx antennas **and all PRBs of the UE group** |
| `2` `PUSCH_EQ_ALGO_TYPE_MMSE_IRC` | **per-PRB full R_ww**, as its inverse Cholesky factor `L_w⁻¹` | sample covariance of the DMRS residual in each PRB |
| `3` `..._MMSE_IRC_SHRINK_RBLW` | per-PRB R_ww after **Rao-Blackwell Ledoit-Wolf** shrinkage | as 2, then shrunk toward a scaled identity |
| `4` `..._MMSE_IRC_SHRINK_OAS` | per-PRB R_ww after **oracle-approximating shrinkage** | as 2, then shrunk toward a scaled identity |

pyAerial exposes the same switch as `eq_coeff_algo`, with default `1` (scalar
MMSE) (`pyaerial/src/aerial/phy5g/config.py:196`).

## 1. Where it sits in the pipeline

`cuphy_channels/pusch_rx.cpp` adds the noise-interference node to the CUDA graph
only when it is needed:

- [`pusch_rx.cpp:2104`] `EqCoeffAlgoIsMMSEVariant()` is true for algorithms
  1–4, false for RZF.
- [`pusch_rx.cpp:2140`] the estimator is set up if
  `enableSinrMeasurement || EqCoeffAlgoIsMMSEVariant(eqCoeffAlgo)`.

So with RZF and SINR reporting off, no noise is estimated at all.

Graph order: **channel estimation → noise-interference estimation → equalizer
coefficient computation → equalization + soft demapping**. There are two
variants of the estimator kernel:
- `CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0` uses only the first DMRS
  symbol (or symbol pair) for the early-HARQ / sub-slot path;
- `CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT` uses all DMRS symbols.

## 2. Noise-interference estimation (`cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu`)

Full-slot kernel `noiseIntfEstKernel`, from line 1164. One thread block covers a
group of PRBs of one UE group, and each PRB is processed separately.

**Step 1: regenerate the transmitted DMRS** for every layer of the UE group, as
in 38.211 §6.4.1.1:
- the Gold scrambling sequence, with c_init built from slot, symbol,
  N_ID and n_SCID;
- the frequency OCC (odd ports) and time OCC (ports 4–7);
- the DMRS power scaling: amplitude √2 when `nDmrsCdmGrpsNoData = 2`, else 1
  ([L1325]);
- for DFT-s-OFDM (`enableTfPrcd = 1`), the low-PAPR Zadoff-Chu sequence
  instead.

A layer contributes only on the subcarriers of its own CDM grid (`gridEnabled`).

**Step 2: residual on the DMRS REs** ([L1664]). For each Rx antenna r,
subcarrier k and DMRS symbol l:

    w[r,k,l] = y[r,k,l] − Σ_layers  Ĥ[r,layer,k,l] · x_dmrs[layer,k,l]

Ĥ is the channel-estimator output of **the same slot**, at the DMRS symbol's
channel-estimate time index.

**Step 3: per-PRB sample covariance** ([L1674] onward). The covariance is
accumulated over the 12 subcarriers of the PRB and the `nDmrsSyms` DMRS
symbols:

    R_ww(prb) = (1/T) Σ_{k ∈ PRB, l ∈ DMRS}  w[:,k,l] w[:,k,l]^H

| `nDmrsCdmGrpsNoData` | subcarriers used | T (samples per PRB) |
|---|---|---|
| 2 | all 12 (both CDM grids, no data on either) | 12 · nDmrsSyms |
| 1 | the 6 tones of grid 0 (grid 1 carries data) | 6 · nDmrsSyms |

- If `noiseCovFlag` is false (algorithm 1, or SINR reporting only), only the
  **diagonal** of R_ww is accumulated.
- For algorithms 2–4 the **full N_rx × N_rx** matrix is accumulated.

**Step 4: diagonal loading** ([L1767]). `CUPHY_NOISE_REGULARIZER = 1e-8`
(`cuphy.h:353`) is added to every diagonal element. Then trace(R_ww) is summed
into a per-UE-group accumulator.

**Step 5, algorithms 3/4 only: shrinkage** ([L1799]–[L1826]). Shrinkage fights
the small-sample problem: T = 12 · nDmrsSyms samples estimate an N_rx × N_rx
matrix. That is only 24 samples for 16 antennas with 2 DMRS symbols, so the raw
sample covariance is badly conditioned. With `tr = trace(R_ww)`,
`tr2 = trace(R_ww R_ww)`, and N = N_rx:

    RBLW:  ρ = clip[0,1]( ((T−2)/T · tr2 + tr²) / ((T+2) · (tr2 − tr²/N)) )
    OAS:   ρ = clip[0,1]( ((1−2/N) · tr2 + tr²) / ((T+1−2/N) · (tr2 − tr²/N)) )

    R_ww ← (1−ρ) · R_ww + ρ · (tr/N) · I

**Step 6, algorithms 2–4: inverse Cholesky factor** ([L1840]). The kernel
factors `R_ww = L_w L_w^H` per PRB and stores `L_w⁻¹` in `tInfoLwInv`, with
shape `(N_rx, N_rx, N_PRB)`. That tensor is the equalizer's input for IRC.

**Step 7: scalar noise variance.** The last thread block of the UE group
computes it ([L1874]–[L1876]):

    σ² = Σ_PRB trace(R_ww(prb)) / (N_rx · N_PRB)
    invNoiseVarLin = 1 / σ²                       → used by algorithm 1
    noiseVarForDtx = σ²                           → used for DTX detection
    noiseVarPreEq  = 10·log10(σ²) + 0.5 dB        → reported (tInfoNoiseVarPreEq)

The **+0.5 dB** (`NOISE_EST_CORRECTION_DB`, [L1321]) compensates for "noise
filtered out by the channel interpolation filter": Ĥ was fit to the same noisy
pilots, so the residual underestimates σ². The correction is applied **only to
the reported dB value**. `invNoiseVarLin`, which the equalizer uses, is the
uncorrected linear value.

## 3. How the equalizer uses it (`cuphy/channel_eq/channel_eq.cu`)

The transmitted symbol covariance is assumed to be the identity (R_xx = I,
`ES_INV = 1`).

**RZF and scalar MMSE** (`eqMmseCoefCompLowMimoKernel` family). The kernel
picks the regulariser at, for example, [L1359]–[L1360]:

    invDiagReg = (eqCoeffAlgo == NOISE_DIAG_MMSE) ? invNoiseVarLin : INV_ZF_REGULARIZER

    M   = H^H · invDiagReg                 (= H^H R_ww⁻¹ with R_ww = σ² I)
    G   = M H + I                          (enhanced Gram matrix)
    Ree = G⁻¹                              (residual error covariance)
    C   = Ree · M                          (MMSE coefficients)

`INV_ZF_REGULARIZER = 3981.07` (`channel_eq_types.cuh:37`) is 10^3.6. RZF is
therefore the same MMSE formula with a fixed σ² = −36 dB: nearly ZF, and
independent of the actual SNR.

**MMSE-IRC, with or without shrinkage** (`eqMmseIrcCoefCompLowMimoKernel`,
L2823, selected at L3339). This kernel whitens with the per-PRB `L_w⁻¹`:

    N   = L_w⁻¹ H                          (whitened channel)
    G   = N^H N + I                        (= H^H R_ww⁻¹ H + I)
    Ree = G⁻¹
    C   = Ree · N^H L_w⁻¹                  (= Ree H^H R_ww⁻¹)

**Bias correction and LLR scaling** (both paths). With
`λ = 1/(1 − Ree_ii)`, capped at `MAX_BIAS_LIMIT = 10` (L67):
- the coefficients are scaled by λ (an unbiased MMSE estimate);
- `ReeDiagInv = 1/(λ · Ree_ii)` is written out. That is the post-equalization
  SINR, which the soft demapper uses to scale LLRs.

## 4. Practical consequences

- **Algorithm 1 uses one σ² per UE group for the whole allocation.** σ² is
  averaged over all antennas and PRBs. Interference that is frequency-selective
  or concentrated on some antennas is smeared out. IRC (2–4) keeps a separate
  R_ww per PRB.
- **The residual includes channel-estimation error.** `w = y − Ĥx` with Ĥ fit
  to the same pilots. The fit absorbs part of the noise (underestimating σ²,
  only partly offset by the +0.5 dB on the reported value). Any model mismatch
  in Ĥ shows up as extra "noise".
- **What counts as interference depends on the CDM layout.** With
  `nDmrsCdmGrpsNoData = 2`, the residual is also taken on the tones of a CDM
  grid that carries no layer of this UE group, where `w = y`.
  - Empty grid: those tones are pure noise.
  - Grid carrying another UE group's DMRS: that DMRS is counted as
    interference, which is the intended behaviour for IRC.
  - Co-scheduled layers that should be cancelled, not counted, must therefore
    be in the **same UE group**.
- **IRC is sample-starved.** T = 12 · nDmrsSyms (or half that) samples per PRB
  estimate an N_rx × N_rx covariance. For large arrays the raw matrix is
  ill-conditioned; shrinkage (3/4) is the remedy, plus the 1e-8 diagonal
  loading.
- **The early-HARQ path** estimates from the first DMRS symbol only, so T is
  halved again when two DMRS positions are configured.

## 5. Relevance to the channel-estimate cache experiment

`eval_4x2_cache/` studies reusing a cached Ĥ across slots. The cuPHY noise
estimator computes exactly the same residual as the cache check,
`E = Y_dmrs − X_dmrs Ĥ`. If Ĥ is a cached estimate, the R_ww or σ² it produces
includes the cache-ageing error, not just noise and interference.

That has two consequences:

1. **A cache test normalised by that σ² would always pass.** The noise-normalised
   test `ε = ‖E‖²/(N σ²)` needs a σ² from an independent source. Computed from
   the same residual, it is ≡ 1 by construction.
2. **The cached receiver should cache σ² or R_ww together with Ĥ.** Take them
   from the residual of the slot in which Ĥ was freshly estimated, and refresh
   them whenever Ĥ is re-estimated. This matches how `invNoiseVarLin` and
   `L_w⁻¹` are already per-slot outputs that only depend on Ĥ.

`scripts/chest_cache.py` currently sidesteps both points by using the exact
simulation σ² (see `eval_4x2_cache/README.md`, Caveats).

[`pusch_rx.cpp:2104`]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy_channels/pusch_rx.cpp
[`pusch_rx.cpp:2140`]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy_channels/pusch_rx.cpp
[L1321]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1325]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1664]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1674]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1767]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1799]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1826]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1840]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1874]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1876]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu
[L1359]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/channel_eq/channel_eq.cu
[L1360]: ../../aerial-cuda-accelerated-ran/cuPHY/src/cuphy/channel_eq/channel_eq.cu
