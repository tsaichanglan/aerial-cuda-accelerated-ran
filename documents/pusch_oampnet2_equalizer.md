# OAMP-Net2 PUSCH Equalizer (`PUSCH_EQ_ALGO_TYPE_OAMPNET2`, `_OAMPNET2_WHITE`)

OAMP-Net2 is an iterative, model-driven MIMO detector. It can be selected as the
PUSCH equalizer through the `eqCoeffAlgo` static parameter, with one of two noise
models (see [Noise models](#noise-models)). It replaces the
soft-demap stage of `PuschRx` with one fused CUDA kernel. That kernel detects every
data resource element (RE) and writes LLRs in the soft demapper's layout, so the
LDPC chain downstream is unchanged.

> **Status:** not yet built in the Aerial container and not yet validated on cuPHY
> test vectors. See [Validation status](#validation-status).

## Contents

- [Selecting the equalizer](#selecting-the-equalizer)
- [Noise models](#noise-models)
- [Why OAMP-Net2 cannot use the MMSE coefficient split](#why-oamp-net2-cannot-use-the-mmse-coefficient-split)
- [Pipeline](#pipeline)
- [End-to-end workflow](#end-to-end-workflow)
- [Kernel design](#kernel-design)
- [Noise variance, detector output and LLR scaling](#noise-variance-detector-output-and-llr-scaling)
- [Integration in `PuschRx`](#integration-in-puschrx)
- [Supported configurations](#supported-configurations)
- [Design notes](#design-notes)
- [Modified files](#modified-files)
- [Validation status](#validation-status)

## Selecting the equalizer

| Entry point | Coloured noise / interference | White noise |
|---|---|---|
| C API (`cuphyPuschStatPrms_t::eqCoeffAlgo`) | `PUSCH_EQ_ALGO_TYPE_OAMPNET2` (`5`) | `PUSCH_EQ_ALGO_TYPE_OAMPNET2_WHITE` (`6`) |
| cuPHY examples (TV config) | `eqCoeffAlgoIdx: 5` | `eqCoeffAlgoIdx: 6` |
| cuPHY-CP | PUSCH `eqCoeffAlgo` index `5` | index `6` |
| pyAerial (`PuschEqCoefAlgoType`) | `PUSCH_EQ_ALGO_TYPE_OAMPNET2` | `PUSCH_EQ_ALGO_TYPE_OAMPNET2_WHITE` |

The cuPHY-CP driver passes the index through unchanged.

Trained parameters are loaded from a binary weight file that the environment
variable `CUPHY_PUSCH_OAMPNET2_WEIGHTS` points to:

```bash
python export_oampnet_weights.py <pickled train_oampnet.py weights> -o model.oampnet2
export CUPHY_PUSCH_OAMPNET2_WEIGHTS=$PWD/model.oampnet2
```

If the variable is not set, `PuschRx` uses untrained parameters
(γ = θ = φ = 1, ξ = 0, which is classical OAMP) with T = 4, and logs a warning.
`puschRxOampNetEq::setWeights()` can switch weight sets between slots, for example
per MCS.

Weight file format (little endian):

```
char[8] "OAMPNET2" | uint32 version (=1) | uint32 numIter T | uint32 nStreams
float32 gamma[T*nStreams] | theta[...] | phi[...] | xi[...]
```

`nStreams = 1` selects the shared (paper) model. `nStreams > 1` gives per-stream
parameters, where parameter k applies to UE-group layer k.

## Noise models

OAMP-Net2 needs a noise model: the noise covariance R enters its error-variance
estimate, its linear estimator and its output variance (see
[Does OAMP-Net2 need the noise variance?](#does-oamp-net2-need-the-noise-variance)).
The two algorithm values differ only in where R comes from and how the kernel
applies it. Everything else (kernel, weights, LLR output) is shared.

| | `OAMPNET2` (coloured) | `OAMPNET2_WHITE` (white) |
|---|---|---|
| Noise model | R = Rww, full Nr×Nr noise-plus-interference covariance per PRB | R = σ²I, one scalar σ² per UE group |
| Front end runs as | `MMSE_IRC_SHRINK_OAS` | `NOISE_DIAG_MMSE` |
| Noise estimator output used | `tInfoLwInv`: LwwInv = chol(Rww_shrunk)⁻¹ per PRB | `invNoiseVarLin` = 1/σ² |
| Whitening in the kernel | `Hw = LwwInv·H`, `yw = LwwInv·y` | `Hw = H/σ`, `yw = y/σ` |
| Covariance shrinkage | OAS shrinkage, before OAMP-Net2 (in the noise estimator) | not needed: no covariance is estimated |
| coefComp (reporting only) | IRC kernel | diagonal MMSE kernel |
| Fallback soft demapper | MMSE-IRC (OAS) | MMSE (diagonal) |
| Matches the trained model | approximately (v² is estimated in the whitened domain) | exactly: same R = σ²I model as training |
| Use when | inter-cell or multi-user interference, correlated noise | thermal-noise-limited links, or to reproduce training conditions |

Why OAS shrinkage for the coloured model: Rww is a sample covariance from only
12 × nDmrsSyms residuals per PRB. With 16 Rx antennas and one DMRS symbol that is
12 samples for a 16×16 matrix, so the raw estimate is rank-deficient and its inverse
Cholesky factor would amplify the weakest directions. OAMP-Net2 is sensitive to
this, because its v² estimate subtracts exactly Nr, which assumes perfect whitening.
OAS replaces Rww by `(1 − ρ)·Rww + ρ·(tr(Rww)/Nr)·I`, with ρ chosen from the data,
before the Cholesky factorization. The shrinkage happens in the noise/interference
estimator; OAMP-Net2 consumes the resulting LwwInv unchanged.

The kernel selects the whitening mode per UE group from the derived `eqCoeffAlgo`
(`NOISE_DIAG_MMSE` means white). Both modes give identical LLRs when
`LwwInv = I/σ`; this was checked numerically (see
[Validation status](#validation-status)).

## Why OAMP-Net2 cannot use the MMSE coefficient split

The MMSE path splits equalization in two stages:

1. **coefComp** computes the equalizer W once per tone, at each DMRS instance.
2. **softDemap** linearly interpolates **W** between DMRS instances (TDI), applies it
   to every data symbol, rotates the result by the CFO, and demaps.

OAMP-Net2 cannot be split like this:

- Its linear stage depends on the error variance v_t², which is estimated from the
  residual `y − H·x_t` of *that* RE.
- Every layer is followed by a non-linear denoiser.

The only thing that could be precomputed per tone is G = HᴴH. The detector therefore
needs the channel estimate at every data RE. It interpolates **H** in time and then
detects, instead of interpolating W.

## Pipeline

```
MMSE (existing)
  chEst (all DMRS instances, per tone)
    -> noise/intf est -> [CFO est]
    -> coefComp: W at each DMRS instance
    -> softDemap: interp(W) * y, CFO rotation, demap -> LLR

OAMPNET2 (coloured)
  chEst (all DMRS instances, per tone)
    -> noise/intf est (full Rww -> OAS shrink -> LwwInv) -> [CFO est]
    -> coefComp (IRC; only feeds post-EQ SINR / Ree reporting)
    -> OAMP-Net2 soft demap, per data RE:
         H_eff = interp_t(H) * CFO rotation
         Hw = LwwInv * H_eff,  yw = LwwInv * y
         T unfolded OAMP-Net2 layers
         max-log LLRs -> LLR

OAMPNET2_WHITE
  chEst (all DMRS instances, per tone)
    -> noise/intf est (diagonal -> scalar 1/sigma^2) -> [CFO est]
    -> coefComp (diagonal MMSE; only feeds post-EQ SINR / Ree reporting)
    -> OAMP-Net2 soft demap, per data RE:
         H_eff = interp_t(H) * CFO rotation
         Hw = H_eff / sigma,  yw = y / sigma
         T unfolded OAMP-Net2 layers
         max-log LLRs -> LLR
```

- **Frequency.** Channel estimation already produces H for every subcarrier
  (`tInfoHEst`, layout `(N_BS_ANTS, N_LAYERS, NF, NH)`). No extra frequency
  interpolation is needed.
- **Time (TDI).** The kernel interpolates H linearly between DMRS instances for each
  data symbol. It uses the same index and weight rule as the MMSE soft demapper
  (`eqMmseSoftDemapKernel_v4`), applied to H instead of W. With TDI disabled, DMRS
  instance 0 is used, the same as the MMSE path.
- **CFO.** coefComp designs W on `H_d · cfo(dmrsSym_d)` for d > 0, and softDemap
  rotates the equalized symbol by `cfo(sym)`. Since |cfo| = 1, this is equivalent to
  detecting on `H_eff = H_d · cfo(dmrsSym_d)[d>0] · conj(cfo(sym))`. The kernel folds
  exactly this rotation into H_eff.
- **Whitening.** The kernel works in the whitened domain, so R = I. Coloured model:
  `Hw = LwwInv·H` and `yw = LwwInv·y`, where `LwwInv` is the lower-triangular inverse
  Cholesky factor of the OAS-shrunk per-PRB covariance (the same tensor the IRC
  coefComp reads). White model: `Hw = H/σ` and `yw = y/σ`, with σ² the UE group's
  scalar noise variance.

## End-to-end workflow

This section follows one slot through `PuschRx` with
`eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_OAMPNET2` (coloured) or `_OAMPNET2_WHITE`, from
the received resource grid to decoded transport blocks. Stages 1–4 are the existing
MMSE front end, unchanged (MMSE-IRC with OAS shrinkage, or diagonal MMSE).
Stage 5 is the OAMP-Net2 kernel, which replaces the soft demapper. Stage 6 is the
existing back end, unchanged.

```
 DataRx (NF_cell, 14, Nr)            received resource grid, fp16
   |
   | 1. Channel estimation (multistage, per DMRS instance)
   v
 tInfoHEst (Nr, Nl, NF, NH)          H per tone, per DMRS instance
   |
   | 2. Noise/interference estimation
   v
 coloured: tInfoLwInv (Nr, Nr, nPrb) LwwInv = chol(OAS-shrunk Rww)^-1 per PRB
 white:    invNoiseVarLin            1/sigma^2 per UE group
   |
   | 3. CFO / TO estimation          (if enabled and >= 2 DMRS instances)
   v
 tInfoCfoEst (14, nUe)               per-symbol CFO correction factors
   |
   | 4. coefComp (IRC / diagonal)    (reporting only: Ree, post-EQ SINR)
   v
 tInfoReeDiagInv, tInfoEqCoef        not read by OAMP-Net2
   |
   | 5. OAMP-Net2 fused kernel, per data RE:
   |      a. time-interpolate H, fold in CFO      -> H_eff
   |      b. whiten H_eff and y (LwwInv or 1/sigma) -> Hw, yw
   |      c. T unfolded OAMP-Net2 layers          -> r_T, tau_T^2
   |      d. max-log demapping                    -> LLRs
   v
 tInfoLLR (N_LLR, Nl, NF, ND)        fp16 LLRs, soft-demapper layout
   |
   | 6. UCI-on-PUSCH LLR segmentation, descrambling + de-rate-matching,
   |    LDPC decoding, CRC
   v
 transport blocks, CRC flags, UCI payloads
```

### Stages 1–4: front end (unchanged MMSE)

| # | Stage | Reads | Writes | Notes for OAMP-Net2 |
|---|---|---|---|---|
| 1 | Channel estimation | DataRx at DMRS REs | `tInfoHEst`, H for every subcarrier and each DMRS instance (NH = 1 + number of additional DMRS positions) | Frequency interpolation is done here. Time interpolation is left to stage 5. |
| 2 | Noise/interference estimation | DataRx at DMRS REs, `tInfoHEst` | Coloured: `tInfoLwInv` and the scalar noise variance. White: the scalar noise variance only (`invNoiseVarLin`). | Coloured: per-PRB Rww from DMRS residuals, OAS shrinkage, Cholesky inverse. White: diagonal of Rww only, averaged into one σ². See [How the noise covariance is computed](#how-the-noise-covariance-is-computed). The mode follows the UE group's mapped `eqCoeffAlgo`. |
| 3 | CFO / TO estimation | `tInfoHEst` across DMRS instances | `tInfoCfoEst` | Only with `enableCfoCorrection` and at least one additional DMRS position. |
| 4 | coefComp (IRC for coloured, diagonal MMSE for white) | `tInfoHEst`, `tInfoLwInv` or `invNoiseVarLin`, `tInfoCfoEst` | `tInfoEqCoef`, `tInfoReeDiagInv` | OAMP-Net2 ignores both outputs. The node stays enabled so post-EQ SINR and noise reporting keep working. |

### Stage 5: OAMP-Net2 kernel

One launch per heterogeneous (Nr, Nl) configuration. The grid is
`(nMaxPrb, ceil(nMaxDataSym / SPB), nUeGroups)`. Each thread block handles one PRB
× SPB data symbols, and each RE is handled by a tile of Nl threads, where thread k
owns stream k. For every data RE, in kernel order:

1. **Load the whitening matrix** (coloured model only). The block loads LwwInv of
   its PRB into shared memory once. All REs of the block share it. In the white model
   nothing is loaded; the kernel uses `σ⁻¹ = √invNoiseVarLin` of the UE group.
2. **Select the RE.** It picks data-symbol index `dsi` and OFDM symbol
   `symIdx = dataSymLoc[dsi]`, and skips the symbol if it is not in the path's
   symbol bitmask (early-HARQ processes only the first symbols).
3. **Load y.** It reads the Nr received samples at cell tone
   `f + 12·startPrb`, symbol `symIdx`.
4. **Interpolate H in time and fold in the CFO** (`loadHeffColumn`, column k):
   - **TDI:** find the DMRS pair (d0, d1) around `symIdx`, with weights
     `a0 = (t1 − sym)/(t1 − t0)` and `a1 = (sym − t0)/(t1 − t0)`. This is the same
     rule the MMSE soft demapper uses for W. Symbols outside the DMRS span are
     linearly extrapolated. With TDI off, or on the early-HARQ path:
     d0 = 0, a0 = 1, a1 = 0.
   - **CFO:** `rot_d = conj(cfo(sym)) · cfo(dmrsSym_d)` for d > 0, and
     `rot_0 = conj(cfo(sym))`. Off on the early-HARQ path.
   - `H_eff(:,k) = a0·rot0·H(:,k,f,d0) + a1·rot1·H(:,k,f,d1)`
5. **Whiten.** Coloured: `Hw(:,k) = LwwInv · H_eff(:,k)` and `yw = LwwInv · y`.
   White: `Hw(:,k) = H_eff(:,k)/σ` and `yw = y/σ`. After this the noise is white with
   unit variance in both models.
6. **Gram matrix.** Row k of `G = Hwᴴ Hw`, and `tr(G)`. G stays fixed across the
   layers.
7. **Initialize** `x_1 = 0`.
8. **Run the OAMP-Net2 layers, t = 1 … T:**
   1. residual `res = yw − Hw·x_t`, and `‖res‖²`;
   2. error variance `v² = max((‖res‖² − Nr) / tr(G), eps)`;
   3. `B = (v²G + I)⁻¹` by Gauss-Jordan in registers (Nl×Nl, no pivoting);
   4. `P = B·G`, and normalization `s = Nl / tr(P)`;
   5. `u = Hwᴴ·res`, then the linear estimate `r_t = x_t + γ_t·s·(B·u)`;
   6. error variance of r: `τ_t²`, shared or per stream (eq. 27);
   7. if t < T, the denoiser:
      `x_{t+1} = φ_t·(E{x | r_t, τ_t²} − ξ_t·r_t)`, using the PAM posterior mean
      for each of I and Q.
9. **Demap** (after layer T, no denoiser). Max-log LLRs from (r_T, τ_T²) for each of
   I and Q, with scale 1/τ_T², clamped to ±65504 and written as fp16 to
   `tInfoLLR(·, k, f, dsi)`. See
   [LLR computation and scaling](#llr-computation-and-scaling).

Equalization and demapping happen in the same kernel. No equalized symbols are
written to global memory between them.

### Stage 6: back end (unchanged)

`tInfoLLR` has the same layout, sign convention and bit order as the MMSE soft
demapper's output, so everything downstream runs unchanged:

- UCI-on-PUSCH LLR segmentation, when the slot carries UCI;
- descrambling and de-rate-matching, which combine LLRs into each code block's
  circular buffer, including HARQ combining;
- LDPC decoding;
- transport-block and code-block CRC checks, with the results reported to L2.

### Early-HARQ (sub-slot) path

With early-HARQ enabled, stage 5 also runs on the sub-slot path, right after the
front-loaded DMRS has been processed:

- Only symbols in `CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK` are
  processed (`..._MMIMO_EXTRA_DMRS` for mMIMO with the maximum DMRS length = 2).
- H comes from DMRS instance 0 only, with no TDI and no CFO. LwwInv (coloured) or σ²
  (white) comes from the front-loaded-DMRS noise estimate.
- The full-slot path later recomputes stages 1–5 with all DMRS instances.

## Kernel design

Each OAMP-Net2 layer t (whitened domain, Nt = number of layers, Nr = number of Rx
antennas):

```
v2   = max((||yw - Hw x||^2 - Nr) / tr(G), eps)        G = Hw^H Hw
B    = (v2 G + I)^-1                                   Nt x Nt, eigenvalues >= 1
s    = Nt / tr(B G)
r    = x + gamma s B Hw^H (yw - Hw x)
tau2 = (tr(C C^H) v2 + theta^2 tr(W W^H)) / Nt         C = I - theta s B G
                                                       (per-stream variant: k-th diagonal terms)
x    = phi (E{x | r, tau2} - xi r)
```

The LLRs are the max-log LLRs of the final (r, τ²).

- **Small inverse.** The matrix inversion lemma turns the Nr×Nr inverse into an
  Nt×Nt one. Its eigenvalues are ≥ 1, so Gauss-Jordan without pivoting is stable and
  no diagonal loading is needed.
- **Thread mapping.** One thread block covers one PRB × SPB data symbols
  (SPB = 8/Nt, at least 1). Each RE gets a tile of Nt threads, and thread k owns
  stream k. Gauss-Jordan runs in registers, and the pivot row is broadcast with
  shuffles.
- **Denoiser.** A PAM-separable posterior mean over the I and Q components
  (Gray-labelled TS 38.211 QAM).
- **LLR output.** Written to `tInfoLLR` with layout `(N_LLR, N_LAYERS, NF, ND)`,
  cuPHY's sign convention (`ln P0/P1`) and the TS 38.211 bit order.
- **Launch interface.** The kernel signature is the same as the MMSE soft demapper:
  `(StatDescr*, __grid_constant__ puschRxChEqSoftDemapDynDescr_t)`. Its launch
  configs therefore drop straight into `m_chEqSoftDemapLaunchCfgs[path]`.

## Noise variance, detector output and LLR scaling

### Does OAMP-Net2 need the noise variance?

Yes. The textbook OAMP-Net2 (and the TensorFlow reference `OAMPNet2Detector`) uses
the noise covariance R in three places:

| Where | Formula with R |
|---|---|
| Error-variance estimate (eq. 26) | v² = (‖y − Hx‖² − tr(R)) / tr(HᴴH) |
| LMMSE-type linear estimator (eq. 19) | Ŵ = v²Hᴴ(v²HHᴴ + R)⁻¹ |
| Output variance (eq. 27) | τ² = (tr(CCᴴ)v² + θ²tr(WRWᴴ)) / Nt |

The reference model assumes white noise, R = σ²I, and takes σ² as an input.

The cuPHY kernel does not use R directly. It whitens H and y first, so that R = I:

```
coloured (OAMPNET2):        Hw = LwwInv · H_eff    yw = LwwInv · y
white    (OAMPNET2_WHITE):  Hw = H_eff / σ         yw = y / σ
                            =>  R_w = I,  tr(R_w) = Nr
```

After whitening, the formulas above hold with R = I. The noise level is carried
implicitly by the scale of Hw and yw: a weaker noise makes Hw larger.

- The coloured model needs the full noise-plus-interference covariance Rww, through
  LwwInv. The scalar `invNoiseVarLin` is not read.
- The white model needs only the scalar `invNoiseVarLin` (1/σ²) of the UE group.
  LwwInv is neither written nor read.

### How the noise covariance is computed

The PUSCH noise/interference estimator (`pusch_noise_intf_est.cu`) computes the noise
model from DMRS residuals. Steps 1 and 2 are common to both models; the white model
stops after step 2.

1. **Residual on DMRS REs.** For every DMRS RE and Rx antenna, the estimator
   subtracts the received pilot predicted from the channel estimate:
   `e = y_dmrs − Σ_layers H · pilot`.
   The residual therefore contains thermal noise, interference, *and* channel
   estimation error.
2. **Per-PRB sample covariance.**
   Rww(prb) = (1 / (12 · nDmrsSyms)) · Σ e·eᴴ, summed over the 12 tones of the PRB
   and all DMRS symbols. The factor is 2 / (12 · nDmrsSyms) when
   `nDmrsCdmGrpsNoData == 1`; only every other tone is used then.
   **White model** (`NOISE_DIAG_MMSE` front end): only the diagonal of Rww is
   accumulated. Its trace (plus the regularizer below), summed over all PRBs of the
   UE group and divided by `nRxAnt · nPrb`, gives σ². The estimator writes
   `invNoiseVarLin = 1/σ²` into the UE group's parameters on the GPU, and the kernel
   uses `σ⁻¹ = √invNoiseVarLin`.
3. **Regularization** (coloured model). `CUPHY_NOISE_REGULARIZER` (1e-8) is added to
   the diagonal. OAS shrinkage then replaces Rww by
   `(1 − ρ)·Rww + ρ·(tr(Rww)/Nr)·I`, with the adaptive OAS coefficient ρ ∈ [0, 1]
   computed per PRB from tr(Rww) and tr(Rww·Rww).
4. **Whitening matrix** (coloured model). Cholesky factorization of the shrunk
   matrix, Rww = L·Lᴴ, then LwwInv = L⁻¹ (lower triangular), written per PRB to
   `tInfoLwInv`, with layout `(Nr, Nr, nPrb)`.

The kernel loads the LwwInv of its PRB once into shared memory and applies it to H
and y of every RE in the block. On the early-HARQ path the estimator uses only the
front-loaded DMRS (`CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0`), unless front-loaded
DMRS sub-slot processing is enabled.

> Shrinkage matters for the coloured model. With 16 Rx antennas and one DMRS symbol,
> each PRB contributes only 12 samples to a 16×16 covariance, so the raw Rww is
> rank-deficient. OAS shrinkage keeps it well-conditioned before the Cholesky
> factorization. The white model estimates no covariance and needs no shrinkage, but
> it ignores any spatial colour of the interference.

### Detector output

Each unfolded layer t produces a pair (r_t, τ_t²) for every stream k of the RE:

- **r_t** is the output of the linear stage,
  `r = x_t + γ_t · s · B · Hwᴴ (yw − Hw x_t)`.
  OAMP's decorrelation makes the error `r − x` approximately Gaussian, so r can be
  treated as an AWGN observation of the transmitted symbol: `r ≈ x + n`,
  `n ~ CN(0, τ²)`. The normalization `s = Nt / tr(B·G)` sets tr(C) = 0 for
  γ = θ = 1 (unbiased on average over the streams, not per stream).
- **τ_t²** is the variance of that error. The shared model uses one τ² per RE,
  averaged over streams (eq. 27). The per-stream model uses the k-th diagonal terms,
  `[CCᴴ]_kk·v² + θ_k²·[WWᴴ]_kk`. τ² is floored at 5e-13.

For t < T, the denoiser turns (r_t, τ_t²) into the next estimate:
`x_{t+1} = φ_t · (E{x | r_t, τ_t²} − ξ_t · r_t)`. The posterior mean is computed
separately for the I and Q components over the Gray-labelled PAM levels, using
weights `exp(−(r − a)² / τ²)`.

The last layer (t = T) skips the denoiser. Its (r_T, τ_T²) go straight to the LLR
computation. The kernel writes **only the LLRs**. Neither r_T (the equalized symbol,
`tInfoDataEq`) nor τ_T² (post-equalization SINR) is written out; post-EQ SINR and
noise reporting still come from coefComp (`tInfoReeDiagInv`; IRC for the coloured
model, diagonal MMSE for the white model).

### LLR computation and scaling

The model is `r = x + n` with complex noise variance τ². Each real component
(I or Q) therefore sees Gaussian noise of variance τ²/2, and the likelihood of PAM
level a is `exp(−(r − a)² / τ²)`. The max-log LLR of bit b in one component is:

```
LLR_b = ( min_{a : c_b = 1} (r − a)²  −  min_{a : c_b = 0} (r − a)² ) / τ_T²
```

- **Scaling factor:** 1/τ_T², the inverse post-detection error variance of the
  stream at that RE. It plays the same role as the per-RE post-equalization SINR
  (ReeDiagInv) in the MMSE soft demapper. There is no extra constant factor.
- **Constellation:** PAM levels of TS 38.211 square QAM, normalized to unit average
  energy: scale 1/√2 (QPSK), 1/√10 (16QAM), 1/√42 (64QAM), 1/√170 (256QAM).
- **Sign:** cuPHY convention, LLR = ln P(b=0)/P(b=1). A positive LLR favours bit 0;
  this is the negative of the Sionna convention.
- **Bit order:** TS 38.211 §5.1. The I component carries b0, b2, …; the Q component
  carries b1, b3, …. Bit b of component iq goes to LLR index `2b + iq`.
- **Range and type:** clamped to ±65504 (the fp16 maximum, as in cuPHY), then stored
  as fp16 with round-toward-zero.
- **Layout:** `tInfoLLR`, `(N_LLR, N_LAYERS, NF, ND)`, the same as the MMSE soft
  demapper, so descrambling, rate matching and LDPC are unchanged.

## Integration in `PuschRx`

OAMP-Net2 is an add-on to the MMSE pipeline. The graph topology, the descriptor
allocator and the existing kernels are unchanged.

1. **Upstream stages run with an MMSE front-end algorithm.** `expandFrontEndParameters()`
   writes `EqCoeffAlgoFrontEnd(eqCoeffAlgo)` into each derived UE group's
   `eqCoeffAlgo`:

   | Static `eqCoeffAlgo` | Derived (per UE group) | Noise estimator | coefComp |
   |---|---|---|---|
   | `OAMPNET2` | `MMSE_IRC_SHRINK_OAS` | full Rww, OAS shrinkage, writes `LwwInv` | IRC |
   | `OAMPNET2_WHITE` | `NOISE_DIAG_MMSE` | diagonal, writes `invNoiseVarLin` | diagonal MMSE |

   As a result, post-EQ SINR and Ree reporting keep working, and the MMSE soft
   demapper of the same front end is a valid fallback for any slot OAMP-Net2 does not
   support. The kernel reads the derived value to choose its whitening mode.

   `EqCoeffAlgoIsMMSEVariant()` includes both OAMP-Net2 values (via
   `EqCoeffAlgoIsOampNet()`), so the noise-estimation nodes and buffers are enabled.
2. **Construction** (`createOampNetEq`, called after `cuphyCreatePuschRxChEq`):
   - loads the weights;
   - allocates three static descriptors (pinned host and device), one each for the
     full slot, early-HARQ, and early-HARQ with the mMIMO extra DMRS symbol;
   - calls `puschRxOampNetEq::init()`.

   OAMP-Net2 is disabled (the front-end MMSE soft demapper is used) for the O-RAN 7.2e
   split and for `PUSCH_NO_SD_DERATE_MATCHING_FEC`.
3. **Per-slot setup** (`setupOampNetEq`, called after the MMSE soft-demap setups
   succeed). It overwrites `m_chEqSoftDemapLaunchCfgs` for the full-slot path, and
   for the sub-slot path when early-HARQ is enabled.
   - **Graph mode:** the existing `update*Graph()` code pushes the configs into the
     soft-demap nodes with `cuGraphExecKernelNodeSetParams`.
   - **Stream mode:** the same configs are launched directly.
   - **Unsupported slot:** the MMSE configs are kept for that slot, and a debug
     message is logged.
4. **Destruction** (`destroyOampNetEq`) frees the descriptor buffers.

## Supported configurations

| | Supported |
|---|---|
| Rx antennas | 2, 4, 8, 16 |
| Layers per UE group | 1, 2, 4, 8 (≤ Rx antennas) |
| Modulation | QPSK, 16QAM, 64QAM, 256QAM, set per layer |
| Unfolded layers T | 1 … 8 |
| Parameters | shared, or per stream |
| Paths | full slot, early-HARQ (sub-slot) |
| Data types | H / LwwInv fp32, DataRx fp16, LLR fp16 (plus fp32 DataRx under `ASIM_CUPHY_FP32`) |
| TDI, CFO correction | full-slot path only (disabled on the early-HARQ path, as in the MMSE soft demapper) |
| Noise models | coloured (OAS-shrunk Rww whitening), white (scalar σ²), chosen per UE group |

**Falls back to the front-end MMSE soft demapper** (MMSE-IRC with OAS for
`OAMPNET2`, diagonal MMSE for `OAMPNET2_WHITE`) for:

- DFT-s-OFDM (transform precoding);
- data on DMRS symbols (`nDmrsCdmGrpsNoData == 1`);
- any other UE-group configuration not listed above.

**Not provided:** the equalized-symbol debug output (`tInfoDataEq`, used by
`enableDebugEqOutput`). A warning is logged when it is enabled.

## Design notes

- **Two noise models, two front ends.** The noise/interference estimator writes
  `tInfoLwInv` only when it computes the full covariance (`MMSE_IRC*`). For
  `NOISE_DIAG_MMSE` it computes only the scalar `invNoiseVarLin`, and for RZF nothing.
  The coloured model therefore needs an IRC front end, and the white model a diagonal
  one. `puschRxOampNetEq::isSupported()` accepts `MMSE_IRC*` and `NOISE_DIAG_MMSE` in
  the derived UE-group parameters and rejects RZF.
- **Why OAS for the coloured model.** Plain `MMSE_IRC` adds only a 1e-8 diagonal
  regularizer to a sample covariance that can be rank-deficient (12 samples per PRB
  with one DMRS symbol). OAS shrinkage is data-adaptive and keeps LwwInv
  well-conditioned. `MMSE_IRC_SHRINK_RBLW` is the other shrinkage estimator cuPHY
  provides and would also work; switching is a one-line change in
  `EqCoeffAlgoFrontEnd()`. Which of the two is better for OAMP-Net2 has not been
  evaluated.
- **Relation to the TensorFlow reference model.** The white model is the same model
  as training (R = σ²I); the kernel and the reference are then identical. With the
  coloured model, W and τ² are unchanged by whitening, but v² is estimated in the
  whitened domain instead of the raw one, which is arguably the better choice under
  interference.
- **Early-HARQ path.** When the sub-slot soft-demap stage runs, only the front-loaded
  DMRS estimate exists. H of later DMRS instances, and the CFO estimate (which needs
  at least two DMRS instances), must not be read. The kernel therefore disables TDI
  and CFO correction whenever its descriptor's symbol bitmask is not the full-slot
  mask.
- **Descriptor per bitmask.** The early-HARQ symbol bitmask depends on the slot's
  maximum DMRS length, so it can change between slots. Setup selects one of three
  pre-filled static descriptors by pointer. This avoids rewriting a descriptor (and a
  blocking copy) while a graph may still be using it.
- **Fallback safety.** `puschRxOampNetEq::setup()` selects every kernel before it
  writes any launch config. A failure therefore leaves the MMSE configs intact.
  Unsupported-type or unsupported-config results are treated as a fallback. Other
  errors fail the slot with `CUPHY_PUSCH_STATUS_SOFT_DEMAP_SETUP_ERROR`.
- **Host-compiled headers.** `oampnet_eq.hpp` is included from `pusch_rx.hpp`, which
  is compiled by the host compiler. The device code in `oampnet_eq_kernel.cuh` is
  guarded by `__CUDACC__`. The constants and descriptor structs stay visible to host
  code.

## Modified files

All paths are relative to the repository root.

### New files

These were copied into the tree from the standalone `cuphy_oampnet` package. The
first three were then modified for the integration.

| File | Contents | Changes made during integration |
|---|---|---|
| `cuPHY/src/cuphy/channel_eq/oampnet_eq_kernel.cuh` | Device code: OAMP-Net2 layers, PAM-separable denoiser, max-log LLRs, H interpolation and CFO handling | Include guard instead of `#pragma once`. Device code guarded by `__CUDACC__` so host-compiled code can include the header. White-noise mode: `OampNetUeGrpView::whiteNoise` / `noiseScale`; LwwInv is skipped and H, y are scaled by 1/σ. |
| `cuPHY/src/cuphy/channel_eq/oampnet_eq.hpp` | Component `channel_eq::puschRxOampNetEq`: interface, weights struct | Three static descriptors (`N_STAT_DESCRS`), one per soft-demap symbol bitmask. Integration contract documented. |
| `cuPHY/src/cuphy/channel_eq/oampnet_eq.cu` | Kernel wrapper, weight loading, het-config batching, launch configs | TDI and CFO disabled on the early-HARQ path. Descriptor selected by pointer, with no blocking `cudaMemcpy` in setup. Whitening mode set per UE group from the derived `eqCoeffAlgo`, with `noiseScale = √invNoiseVarLin`. `isSupported()` accepts `MMSE_IRC*` and `NOISE_DIAG_MMSE`. `setup()` selects every kernel before it writes any launch config. |
| `cuPHY/src/cuphy/channel_eq/export_oampnet_weights.py` | Converts trained weights into the `.oampnet2` binary | None |

### Modified existing files

| File | Change |
|---|---|
| `cuPHY/src/cuphy/cuphy.h` | New enum values `PUSCH_EQ_ALGO_TYPE_OAMPNET2 = 5` and `PUSCH_EQ_ALGO_TYPE_OAMPNET2_WHITE = 6` in `cuphyPuschEqCoefAlgoType_t`. |
| `cuPHY/src/cuphy/CMakeLists.txt` | Adds `channel_eq/oampnet_eq.cu` to the `cuphy` sources. |
| `cuPHY/src/cuphy_channels/pusch_rx.hpp` | Includes `channel_eq/oampnet_eq.hpp`. New members `m_oampNetEq`, `m_oampNetEnabled`, `m_oampStatCpu`, `m_oampStatGpu`. New methods `createOampNetEq()`, `destroyOampNetEq()`, `setupOampNetEq()`. |
| `cuPHY/src/cuphy_channels/pusch_rx.cpp` | See the list below. |
| `cuPHY/examples/common/datasets.cpp` | `eqCoeffAlgoIdx: 5` maps to `PUSCH_EQ_ALGO_TYPE_OAMPNET2`, `6` to `PUSCH_EQ_ALGO_TYPE_OAMPNET2_WHITE`. |
| `pyaerial/pybind11/pycuphy_pybind.cpp` | Adds `PUSCH_EQ_ALGO_TYPE_OAMPNET2` and `PUSCH_EQ_ALGO_TYPE_OAMPNET2_WHITE` to the `PuschEqCoefAlgoType` enum. |

Changes in `pusch_rx.cpp`:

- `#include <cstdlib>`, needed for `std::getenv`.
- New file-scope helpers `EqCoeffAlgoIsOampNet()` (true for both OAMP-Net2 values)
  and `EqCoeffAlgoFrontEnd()` (OAMPNET2 → `MMSE_IRC_SHRINK_OAS`, OAMPNET2_WHITE →
  `NOISE_DIAG_MMSE`, anything else unchanged).
- `createComponents()` calls `createOampNetEq()` after `cuphyCreatePuschRxChEq()`
  when `EqCoeffAlgoIsOampNet(eqCoeffAlgo)`.
- `destroyComponents()` calls `destroyOampNetEq()`.
- `EqCoeffAlgoIsMMSEVariant()` returns true for both OAMP-Net2 values. This keeps the
  noise/interference estimation nodes and buffers enabled.
- New `PuschRx::createOampNetEq()`:
  - loads the weights from `CUPHY_PUSCH_OAMPNET2_WEIGHTS`, or uses classical OAMP
    with T = 4 if the variable is not set;
  - allocates the static descriptors and calls `puschRxOampNetEq::init()`;
  - stays disabled for the 7.2e split and for `PUSCH_NO_SD_DERATE_MATCHING_FEC`;
  - logs the noise model in use.
- New `PuschRx::destroyOampNetEq()` frees the descriptor buffers.
- New `PuschRx::setupOampNetEq()` overwrites `m_chEqSoftDemapLaunchCfgs` for the
  full-slot path, and for the early-HARQ path when it is enabled. If the slot's
  UE-group configuration is not supported, it keeps the MMSE configs.
- `setupComponents()`:
  - calls `setupOampNetEq()` after the MMSE soft-demap setups succeed;
  - adds a `setupOampNetStatus` check, which reports
    `CUPHY_PUSCH_STATUS_SOFT_DEMAP_SETUP_ERROR` on failure.
- `expandFrontEndParameters()` writes `EqCoeffAlgoFrontEnd(eqCoeffAlgo)` into each
  derived UE group's `eqCoeffAlgo`.

## Validation status

**Done:**

- Standalone kernel parity against the TensorFlow `OAMPNet2Detector`, run before the
  integration changes (i.i.d. Rayleigh, R = σ²I). The CUDA and TF BER matched for
  16x8, 16x4, 8x8, 4x2 and 2x1 antenna/layer configurations, QPSK to 256QAM, and
  shared, per-stream and T = 5 weights.
- After the integration changes:
  - the standalone test driver compiles with nvcc against the modified kernel header;
  - the header's host-visible part compiles with g++ `-Werror`;
  - white-noise mode, standalone driver, 4x2 16QAM, 8 PRB × 12 symbols, σ = 0.3:
    LLRs with `noiseScale = 1/σ` (and LwwInv zeroed) are identical to LwwInv = I/σ
    (9,216 LLRs, max relative error 0, 100% sign agreement). A control run with
    LwwInv zeroed but white mode off gives different LLRs, which confirms that the
    white path does not read LwwInv.

**Pending:**

- Build in the Aerial container (`-Werror`) and run the PUSCH unit tests.
- cuPHY test vectors with TDI, CFO correction and early-HARQ, plus end-to-end BLER
  through LDPC.
- BLER comparison of the two noise models under white noise and under interference.
- Confirm that the full-slot noise estimation cannot overwrite `LwwInv`,
  `invNoiseVarLin` or H while the early-HARQ OAMP-Net2 stage is still reading them. Unlike MMSE, this stage reads
  those tensors directly instead of precomputed coefficients.
- Performance at 273 PRB. Standalone measurements on a loaded RTX 6000 Ada were
  about 486 µs per launch for 16x8, 189 µs for 16x4 and 51 µs for 4x2.
  - Known improvement: share Hw and G across the data symbols of a tone when H_eff
    does not change with the symbol (no TDI, no CFO).

## Estimated time for 4x2

Configuration: 4 Rx antennas, 2 layers, at full bandwidth.

| Item | Value |
|---|---|
| Allocation | 273 PRB × 12 data symbols (39,312 data REs per layer) |
| Data types | H / LwwInv fp32, DataRx fp16, LLR fp16 |
| Kernel time | **about 51 µs per launch** (one OAMP-Net2 soft-demap launch) |
| Launch shape | 96 threads per block (12 tones × 4 symbols × 2 layers), 10 KB shared memory, 56 registers per thread |
| Grid | 273 × 3 × number of UE groups (SPB = 4 data symbols per block) |

How to read this number:

- It is an **upper bound**. It was measured on an RTX 6000 Ada with the standalone
  test driver (`run_parity.py --time`) while the GPU was 99–100% busy with other jobs.
  It was not measured inside the cuPHY pipeline.
- It covers only the OAMP-Net2 kernel, which replaces the MMSE soft demapper. Channel
  estimation, noise/interference estimation and coefComp still run as in the MMSE
  pipeline and add to the total PUSCH time.
- Run time scales roughly linearly with the number of data REs (PRBs × data symbols).
  It also grows with the number of unfolded layers T (default 4), but less than
  linearly: the per-RE whitening and the G = HwᴴHw computation are done once,
  whatever T is.
- It has not been re-measured since the integration changes. It was measured with
  LwwInv whitening (coloured model). The white model skips the Nr×Nr LwwInv
  multiplication per RE and should be slightly faster; this is not measured.
