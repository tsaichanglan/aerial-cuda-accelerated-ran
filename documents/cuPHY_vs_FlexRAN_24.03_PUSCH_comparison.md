# cuPHY vs FlexRAN 24.03: PUSCH receiver comparison

This document compares the PUSCH receive chain of NVIDIA cuPHY (this repository) with Intel FlexRAN 24.03. It covers channel estimation, the Wiener filter, equalizer coefficients, MMSE equalization, demapping and LLRs, noise variance, the noise-plus-interference covariance, and the pre-eq SNR, post-eq SINR, RSRP and RSSI measurements.

## Sources and depth of verification

| Side | Source | Depth |
|---|---|---|
| cuPHY | Code paths traced in [PUSCH_ChEst_ChEq_workflow.md](PUSCH_ChEst_ChEq_workflow.md) | Read in detail |
| FlexRAN | `D:\FlexRAN\v24_03\source\nr5g\docs\FlexRAN_24.03_PUSCH_Uplink_Workflow.md` (Claude-generated) | Checked against the code below |
| FlexRAN kernels | `D:\FlexRAN\v24_03\sdk\source\phy\`: `lib_cestimate_5gnr`, `lib_irc_rnn_calculation_5gnr`, `lib_pusch_symbol_processing_5gnr`, `lib_common/mimo.hpp`, `lib_common/matrix.hpp` | Read directly at the lines cited |
| FlexRAN orchestration | `source/nr5g/gnb_ul`, `api/phy2mac` | Taken from the FlexRAN doc only; not re-verified |

The FlexRAN SDK kernels (`bblib_*`) **are** in the FlexRAN tree, under `sdk/source/phy`, outside `source/nr5g`.

Paths below shortened as `FR/` = `D:\FlexRAN\v24_03\sdk\source\phy\`.

---

## 1. Architecture

| | cuPHY | FlexRAN 24.03 |
|---|---|---|
| Platform | CUDA kernels in CUDA graphs. FP32 compute; FP16 storage for DataRx and LLR | AVX-512 / AVX2 (FP16 on SPR). IQ is int16 Q13; LLRs are int8 |
| Pipeline | ChEst → NoiseIntf → (RSRP) → CFO/TA → EqCoef → SoftDemap, one kernel per stage | CE (including noise, SNR, CFO, Doppler) → R_nn → **one fused kernel** (`bblib_pusch_symbol_processing`): FO compensation, time interpolation, MMSE/IRC and LLR. FR2 uses separate stages |
| Early start | Early-HARQ kernel variants that process the first DMRS symbol only | Two half-slots (CE0/EQ0 and CE7/EQ7) |
| Parallel split | Per UE group; kernels with the same configuration are batched into one launch | eBBU task graph, split by UE group, RB or antenna |

## 1.1 AGC and input scaling

**cuPHY has no AGC.** It has no counterpart to `bblib_agc_detection`, and no per-UE or per-slot gain detection. The only gains are a **fixed per-cell scale** applied when fronthaul IQ is decompressed, and a **dB calibration offset** applied when measurements are reported.

**FlexRAN AGC** (see §3.3 of the FlexRAN doc). `bblib_agc_detection` runs once per UE group, on the first DMRS symbol:

1. Measure the RMS amplitude.
2. Compute g = round(log₂(rms / target)).
3. The CE and equalizer kernels apply Y ≫ g internally.
4. s_AGC = 2^{2g} restores the absolute level of the reported interference-plus-noise power.

It is needed because IQ is int16 Q13: without it, weak signals lose bits and strong signals saturate.

**cuPHY: fixed decompression scale β_ul** (in cuPHY-CP, not cuPHY).

- Set once per cell from the compression configuration ([cell.cpp:753-759](../cuPHY-CP/cuphydriver/src/common/cell.cpp#L753-L759)):
  ```
  sqrt_fs0 = 2^(ul_bit_width−1) · 2^(2^exponent_ul − 1)
  fs       = sqrt_fs0² · 2^(−fs_offset_ul)
  beta_ul  = max_amp_ul / sqrt(fs)                      (BFP / fixed-point compression)
  beta_ul  = 1 / 2^(ul_bit_width − 5)                   (no compression: "3 integer and 1 sign bit")
  ```
- Multiplied by an OAM attenuation gain: `getBetaUlPowerScaling() = beta_ul · oam_linear_gain`, where `oam_linear_gain = 10^(−attenuation_dB/20)` ([cell.cpp:1481-1486](../cuPHY-CP/cuphydriver/src/common/cell.cpp#L1481-L1486)).
- Applied by the UL order kernel while it decompresses each packet straight into the FP16 `DataRx` buffer (`decompress_scale_blockFP` / `decompress_scale_fixed` with `beta_cell`, [order_cuda_kernels.cu:1395-1399](../cuPHY-CP/cuphydriver/src/uplink/order_cuda_kernels.cu#L1395-L1399)).
- **Static**: the same for every UE, slot and PRB, and independent of the received power.

**Why cuPHY needs no AGC.**

- `DataRx` is FP16 and all PUSCH math is FP32. Floating point keeps the same *relative* precision over its range (about 3 significant digits for FP16, from about 6·10⁻⁵ up to 65504), so weak and strong UEs are estimated equally well without a shift.
- The receiver is scale-invariant. σ², R_ww and Ĥ all scale together, so the MMSE coefficients, `ReeDiagInv` and the LLRs do not depend on the absolute level.
- β_ul therefore only has to place the fronthaul full scale well inside the FP16 range, which a fixed per-cell value does.

**Absolute-power calibration.** cuPHY reports RSSI, RSRP and interference in **dB of digital power**. The SCF L2 adapter subtracts the per-cell `ul_gain_calibration` to convert to dBm. For example, `pn_dbm = pInterf − ul_gain_calibration` ([scf_5g_fapi_phy.cpp:4168](../cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4168)); see also `ul_ind_rssi` / `ul_ind_rsrp` (`:4426-4428`). `cuphycontroller_F08.yaml` sets it to 48.68 dB. This plays the role of FlexRAN's `Rx0DbfsRadioPower` calibration (§4.1 of the FlexRAN doc), not of its AGC.

| | FlexRAN | cuPHY |
|---|---|---|
| Why gain control is needed | int16 Q13 fixed point: weak signals lose bits, strong signals saturate | Not needed: FP16 storage, FP32 compute |
| Gain detection | `bblib_agc_detection`, once per UE group, from the first DMRS symbol | None |
| Gain value | g = round(log₂(rms / target)), per UE group, per slot | β_ul: fixed per cell, from compression configuration × OAM attenuation |
| Where applied | Y ≫ g inside the CE and equalizer kernels | Once, during fronthaul decompression into FP16 `DataRx` |
| Restoring absolute level | s_AGC = 2^{2g} for IpN power; `Rx0DbfsRadioPower` converts dBFS to dBm | No per-slot gain to undo; `ul_gain_calibration` converts dB to dBm in the L2 adapter |
| Overflow reporting | `nAgcOverflow` | None found |

**Caveat.** β_ul is fixed, so a misconfigured cell could let a very strong input exceed FP16's maximum (65504), or push a very weak input into the subnormal range where precision is lost. FlexRAN's AGC adapts to both cases. No overflow detection was found on the cuPHY side.

## 2. Channel estimation and the Wiener filter

| | cuPHY | FlexRAN |
|---|---|---|
| LS / OCC | Descramble, then remove tOCC, then fOCC | Same: Y·X*/β, then fOCC/tOCC de-spread |
| Delay centring | Algorithm 0: fixed τ (0.4375 µs). Algorithm 1: τ estimated per UE group from the correlation between pilot tones. Shift before filtering, unshift after | Per-layer TO from the correlation between DMRS REs L subcarriers apart. Shift before the filter, de-compensate after |
| Filter W | **Static** MMSE filter tables (`cuPhyChEstCoeffs.h5`). Fixed windows: read 8 PRBs / write 4, or read 4 / write 2. Separate edge filters. The documented legacy and multi-stage paths have no per-slot SNR selection. RKHS is an alternative estimator | **Precomputed sinc-model Wiener LUT**, indexed by μ, delay-spread index, CDM grid and **SNR from −10 to 40 dB in 0.25 dB steps (201 points)**. Default `nInterpMode = 0` is 1 RB + sinc (6 → 12 for type 1). 4 RB + sinc is an option |
| SNR adaptivity | None | **Two passes:** filter assuming 30 dB, estimate σ² and SNR, then re-filter with the SNR-matched weights (`FR/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp:1387-1409`, `:4204-4250`) |
| Delay-spread adaptivity | Only through τ (the mean delay), not the spread | `n_delay_spread_index` selects the table. Optional per-slot delay-spread estimation |
| Output in time | One Ĥ per DMRS position (`NH`) | One Ĥ per CE symbol |

### 2.1 How FlexRAN selects the Wiener filter

FlexRAN does not compute Wiener weights at run time. It selects a precomputed weight matrix from a lookup table. Four indices choose the matrix, plus an interpolation-mode switch that chooses the table. The SNR and delay-spread indices are measured per slot. The others come from configuration.

```
W = Table[mode][μ][dsIdx][Δ][snrIdx]      (FR/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp:4220-4278)
```

| Index | Values | Where it comes from |
|---|---|---|
| `mode` (table) | 0 = 1 RB + sinc: `gCeType{1,2}WienerCoeffSinc`, 6×12 (type 1) or 4×12 (type 2). 1 = 4 RB + sinc: `gCeType{1,2}WienerCoeff4RbSinc`, 24×48 or 16×48 | `n_interp_method & CE_POST_INTERP`, from XML `CEInterpMethod`. Forced to 0 for resource allocation type 0 with RBG size 2 (`nr5g/api/mac2phy/gnb_mac2phy_api_proc_ul_config.c:527-536`). 4 RB is used only in the AVX-512 INT16 build |
| `μ` | 0…`MAX_MU_NUM`−1 | Cell numerology (`n_mu`) |
| `dsIdx` | 0…4 (`DELAY_SPREAD_NUM = 5`) | Configured, or estimated per slot (see 2.1.2) |
| `Δ` (CDM group) | Type 1: 0–1. Type 2: 0–2 | From the DMRS port: `type1_delta = {0,0,1,1,0,0,1,1}`, `type2_delta = {0,0,2,2,4,4,…}/2` |
| `snrIdx` | 0…200 for −10…40 dB in 0.25 dB steps (`CE_WMATRIX_SNR_POINT = 201`) | Estimated per layer in a two-pass CE (see 2.1.1) |

**Where the tables come from.** The weights are generated offline from a sinc (uniform power-delay-profile) correlation model for each delay spread and SNR. They ship as binary files that `bblib_cestimate_table_init()` loads once at start-up (`phy_cestimate_5gnr_internal.hpp:361-366`, `phy_cestimate_5gnr_simd.cpp:4425-4507`):

- `ce_type1_w_sinc.bin`, `ce_type2_w_sinc.bin`
- `ce_type1_w_4rb_sinc.bin`, `ce_type2_w_4rb_sinc.bin`
- `mmse_coeff_lookup_table.bin` and `mmse_coeff_scaling_factor_lookup_table.bin` (used only by the delay-spread estimator)

The entries are complex int16. The SDK contains no generator for these tables.

#### 2.1.1 SNR index: a two-pass estimate

1. **Pass 1 assumes 30 dB.** The SNR-estimation kernel (`dmrs_mmse_ce_snr_est`) always uses the 1 RB table, at `snrIdx = floor((30 + 0.1249)·4) + 40 = 160` (`:1385-1409`, `:1471`).
   - It filters the LS estimate with W, re-applies fOCC to each layer's Ĥ, sums over the layers of the CDM group, and re-modulates with the DMRS (X·β).
   - The residual is e = Y − Σ Ĥ·X·β, so σ² = Σ|e|² / N_DMRS_SC, times the noise-scaling factor α where enabled (`:1650-1688`).
   - Per-layer signal power: P_S,l = mean|Ĥ_l|²·β²/4 − σ²·noiseFac_l (`:1689-1731`). The β²/4 factor compensates for the fixed-point scaling.
2. **Pass 2 uses each layer's own SNR** (`:4204-4250`):
   - SNR_l = P_S,l / σ², clamped to −10…40 dB.
   - `snrIdx = floor((SNR_dB + 0.1249)·4) + 40`. The 0.1249 offset rounds to the nearest 0.25 dB grid point; the +40 maps −10 dB to index 0.
   - W is fetched per layer, using that layer's Δ and SNR, and applied to the full allocation by `dmrs_mmse_ce_post_interp_sinc_5gnr` (or the 4 RB variant), with TA shift and unshift.

The noise-scaling factor α = 1/(1 − Σ noiseFac) uses a separate, coarser lookup (`pnoiseFac_dmrsType{1,2}`):

- an 11-level SNR bucket, with linear thresholds 0.1778, 0.5623, …, 5623 (−7.5 to 37.5 dB in 5 dB steps)
- delay-spread class `(dsIdx & 7) >> 2`, so it only separates index 4 from indices 0–3
- layers in the CDM group (1 or 2)
- μ (0–2)

Pass 1 uses bucket 8 (30 dB); pass 2 uses the bucket of the measured SNR (`:4151-4197`). Note: the shipped L1 sets `p_ce_req->noiseScaling = 0` (`nr5g/gnb_ul/gnb_ul_phy_pusch.c:1163`), so in that path α = 1 and noiseFac = 0. The noise-scaling feature exists in the kernel but is off.

#### 2.1.2 Delay-spread index: configured or estimated

**Configuration** (`nr5g/api/mac2phy/gnb_mac2phy_api_proc_ul_config.c:495-545`):

- MMSE CE: `n_delay_spread_index = 4`, which the code comments as "300 ns delay spread by default".
- If XML `nEnableDelaySpreadEst = 1` and the DMRS is type 1, it is set to **5**, which means "estimate it".
- DCT CE, used when there are more layers than OCC ports: index 0 and no estimation.

**Run-time gate** (`phy_cestimate_5gnr_simd.cpp:4016-4033`). In the AVX-512 INT16 and FP16 builds:

- if the index is 5, nRB ≥ `CE_MIN_PRB_DS` (2) and the DMRS is single-symbol, `delay_spread_estimtion_5gnr()` runs
- otherwise the index falls back to 4

The chosen value is returned in `p_ce_resp->n_est_delay_spread_idx`. The AVX2 build passes the configured index through unchanged. The estimate is one value per CE call (UE group and slot), shared by all layers and Rx antennas.

**The estimator** (`delay_spread_estimtion_5gnr`, `:3523-3781`):

1. **Aggregated channel.** For each CDM group and layer, it gathers the LS estimates of the first `CE_MAX_PRB_DS` = 48 PRBs (or fewer) into one vector per Rx antenna. A single-layer UE produces two aggregated channels.
2. **Correlation-based SNR and timing offset.** For each aggregated channel and Rx antenna:
   - P = Σ|h_k|² and c = Σ h_{k+1}·h_k*
   - SNR = |c| / |P − |c||
   - the summed correlation gives TO = −N_FFT·∠c / (2π·intervalRe), with intervalRe = 4 (type 1) or 6 (type 2)
   - TO is clamped to ±CP and removed from every aggregated channel by a phase ramp (`:3611-3681`)
3. **MMSE-prediction hypothesis test.** There are 12 delay-spread hypotheses, k = 0…11, with DS_k = CP_len·(23/32)^(11−k), from about 0.026·CP to 1.0·CP.
   - Each hypothesis has a 4-tap MMSE predictor from `gDsEstMmseCoeff`, indexed by DMRS RE spacing (type 1 or 2), k, and SNR bucket `round(SNR_dB/5)` clamped to 0…6 (0…30 dB).
   - Each channel sample is predicted from its 4 neighbours. The prediction-error power (MSE) is summed over Rx antennas and aggregated channels.
   - The hypothesis with the smallest total MSE wins (`:3707-3756`).
4. **Coarse to fine.**
   - Pass 1 tests k ∈ {1, 4, 7, 10}.
   - It then refines the SNR: σ² = MSE·scale, with scale from `gDsEstMmseCoeffScaFact` in Q15, and SNR = (P − σ²)/σ².
   - Pass 2 tests {k̂−1, k̂, k̂+1} with the refined SNR buckets (`:3757-3778`).
5. **Map to a table index** (`map_legacy_DS`, `:3496-3518`): DS = CP_len(ns)·ratio[k̂] is snapped to the nearest of {0, 54, 200, 400, 600}. The code comments these as "legacy DS × 2", i.e. about 0, 27, 100, 200 and 300 ns RMS delay spread, giving `dsIdx` 0…4.

#### 2.1.3 Summary of what drives the filter choice

| Quantity | How it is obtained | Granularity |
|---|---|---|
| SNR | Pass 1 residual after a 30 dB filter, then P_S/σ² per layer | Per layer, per DMRS symbol group, per slot |
| σ² | Re-modulated residual over the DMRS REs, all Rx antennas | Per CDM group |
| Delay spread | Fixed at 4 (300 ns), or the MMSE-prediction hypothesis test above | Per UE group, per slot (type 1, single-symbol DMRS, ≥ 2 PRB) |
| Timing offset | Correlation phase; shifted out before filtering. Not a table index | Per layer (CE), or per UE group (DS estimator) |
| Numerology, CDM group, mode | Configuration and DMRS port | Static or per allocation |

**Contrast with cuPHY.** cuPHY uses one fixed set of MMSE filters (`cuPhyChEstCoeffs.h5`), with no SNR or delay-spread index. It adapts only the delay centring τ (fixed in algorithm 0, estimated per UE group in algorithm 1). FlexRAN's SNR selection is closed loop within the slot. Its delay-spread selection is optional and off by default.

## 3. Time interpolation and CFO

| | cuPHY | FlexRAN |
|---|---|---|
| What is interpolated | **Equalizer coefficients C** (TDI). Linear between the two bracketing DMRS positions, extrapolated at the ends, up to 4 coefficient sets. Without TDI, one set serves the whole slot | **Channel Ĥ**: nearest (default), linear, Lagrange or grouped-linear. The granularity is chosen from Doppler/SNR tables. W is then recomputed for each channel symbol |
| CFO correction | Equalized symbol de-rotated: z·CfoEst[s]. The IRC coefficient kernel also rotates H at later DMRS positions | **Channel** rotated (H·e^{−jφl_j} → interpolate → ·e^{+jφl}), because the REs contain several UEs |

### 3.1 More than two DMRS symbols: what is interpolated, and where the matrix inverse is computed

**cuPHY interpolates the equalizer coefficients. FlexRAN interpolates the channel and recomputes the MMSE inverse for each interpolated channel.**

**cuPHY.**

- Channel estimation produces one Ĥ per DMRS position, up to 4 with `dmrsAddlnPos` = 3 (`CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST`, [cuphy.h:71](../cuPHY/src/cuphy/cuphy.h#L71)). Ĥ itself is never interpolated in time.
- **TDI on** (`enablePuschTdi` = 1). The coefficient kernel runs once per DMRS position ([pusch_rx.cpp:4211-4219](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L4211-L4219)), giving up to 4 sets of C and `ReeDiagInv`. That is the expensive LDL solve. The soft-demap kernel then interpolates **C** linearly between the two DMRS positions that bracket each data symbol, and extrapolates at the ends. The per-symbol cost is one multiply-add per coefficient.
- **TDI off.** Only **one** coefficient set is computed (`bound = 1`), from the **first DMRS position's** Ĥ, and it is used for every data symbol. The additional DMRS symbols still feed noise estimation, CFO/TA and RSRP. The CFO de-rotation corrects the phase drift. Amplitude and Doppler variation across the slot are not tracked.
- CFO: the IRC kernel removes the CFO phase from Ĥ at later DMRS positions, so all coefficient sets share one phase reference. The equalized symbol is then de-rotated: z·CfoEst[s].

**FlexRAN** (fused kernel `bblib_pusch_symbol_processing`, `FR/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr_avx512.cpp`).

- **Interpolation on** (`nLinInterpEnable` = 1). There are separate template instances for 2, 3 and 4 CE symbols per half-slot (`:1382-1401`). `init_table_interp` (`:588-649`) groups the data symbols:
  1. Split the data symbols into groups of `groupSize[nGranularity]` = {1, 2, 3, 4, 6} consecutive symbols (`FR/lib_equalization/phy_rx_mimo_mmse_avx512.cpp:49`).
  2. For each group, take the **mean symbol index**. Find the two CE symbols that bracket it and compute the linear weights w₀, w₁ for that position.
  3. Flag the group's **first** symbol in `flag_symH_upd`.
- On each flagged symbol (`:766-826`):
  1. Interpolate the channel: Ĥ(l) = w₀·Ĥ[ce−1] + w₁·Ĥ[ce]. Weights come from `wSym`, or from the tabulated per-layer `pIntrpweights` at granularity ≥ 6.
  2. Build A = ĤᴴĤ + σ²I, do a **full `matrix_inverse<N_TX>`**, and compute β.
  3. The remaining symbols of the group reuse that inverse and those gains.
- Before interpolating, the CFO phase is removed from Ĥ at later DMRS symbols (`fmulconj(…, FoOffsetTable)`, `:729-734`).
- **Interpolation off.** **Nearest DMRS**: one inverse per CE symbol, serving its nearest data symbols (`nSymbPerDmrs`, `:454-491`). Every DMRS symbol is still used.
- **Frequency.** Both stacks compute coefficients for every subcarrier. FlexRAN processes 16 subcarriers per SIMD step, one per lane, each with its own inverse, with no sharing across subcarriers or RBs in the MMSE path.
- The kernel only ever combines **two** bracketing CE symbols. The three-point "Lagrange (1+1+1)" option described in the FlexRAN doc (§3.7) was not found in this kernel.

**Matrix inversions per subcarrier per slot** (example: 12 data symbols, 2 DMRS positions):

| Setting | FlexRAN | cuPHY |
|---|---|---|
| Granularity 1 (every data symbol) | **12** | — |
| Granularity 2 / 3 / 4 / 6 | 6 / 4 / 3 / 2 | — |
| Interpolation off | 1 per CE symbol (nearest) | 1 (first DMRS only) |
| cuPHY TDI on | — | **1 per DMRS position** (2 here, up to 4), then C is interpolated linearly |

The gnb table `nMldRSymbNum` gives the matching group counts: {12, 6, 4, 3, 2} for 2 DMRS positions and {10, 5, 4, 3, 2} for 4. The granularity is chosen per slot from Doppler and SNR. At low Doppler FlexRAN typically uses granularity 6, which costs about the same as cuPHY. At high Doppler it can drop to one inversion per data symbol.

**Trade-off.**

| | cuPHY | FlexRAN |
|---|---|---|
| What is interpolated | Equalizer coefficients C | Channel Ĥ |
| Heavy solve happens | Only on DMRS positions (≤ 4 per slot) | Once per symbol group (2 to 12 per slot for 2 DMRS) |
| Accuracy between DMRS | Approximate, because C is not linear in H | Exact MMSE inverse for the interpolated channel |
| Cost | Low, and fixed regardless of Doppler | Tunable through granularity, highest at high Doppler |
| Without interpolation | First DMRS only, for the whole slot | Nearest DMRS for each data symbol |

The FlexRAN doc says IRC mode `nEnableIrc == 2` does "2-SC processing". Whether that also shares the equalizer inverse between pairs of subcarriers was not checked.

## 4. Noise variance

| | cuPHY | FlexRAN |
|---|---|---|
| Residual | e = Y − Σ Ĥ·x on DMRS REs (all 12 subcarriers when both CDM groups carry no data) | e = Y − Σ Ĥ·β·x |
| σ² | mean tr(R_ww) / (nRx·nPrb). The linear value given to the equalizer (`invNoiseVarLin`) has **no degrees-of-freedom correction** | mean\|e\|² × **1/(1 − Σ noiseFac)**. noiseFac comes from a LUT by delay spread, layers, SCS and SNR bucket, and accounts for the noise the filter removed. Applied only when μ ≤ 2 and ≤ 2 layers per CDM group (`phy_cestimate_5gnr_simd.cpp:1661-1686`, `:4151-4197`) |
| Report correction | +0.5 dB added only in the dB report (`NOISE_EST_CORRECTION_DB`) | The corrected σ² is used everywhere, including the equalizer |
| Extra scaling | — | `PuschNoiseScale` × σ² for multi-layer single-UE at μ = 0 (gnb_ul) |

## 5. Noise-plus-interference covariance (IRC)

| | cuPHY | FlexRAN |
|---|---|---|
| Granularity | **Per PRB**, averaged over all DMRS symbols | **Per 4-RB RBG and per DMRS symbol** (optionally averaging the two symbols) |
| Regularization | ε = 1e-8 on the diagonal | — |
| Shrinkage | Adaptive **RBLW / OAS**: ρ from tr(R) and tr(R²) of the sample covariance | Fixed **α = 0.25** (`N_REFINE = 1`), or α = a·INR_dB + b from a table by nRx and INR (`N_REFINE = 2`). Same form: (1 − α)R + α·tr(R)/N·I (`FR/lib_irc_rnn_calculation_5gnr/phy_irc_rnn_calculation_5gnr_avx512.cpp:45-85`, `:256-273`) |
| Inversion | Cholesky, then **L⁻¹ only**. R⁻¹ is never formed; it is applied by whitening N = L⁻¹H | **Explicit R_nn⁻¹** stored per RBG (`rnn_inverse_all`), then multiplied as Hᴴ·R_nn⁻¹ (`HTransMulInvRnn2`, `FR/lib_common/mimo.hpp:360`) |
| MMSE/IRC choice | Static `eqCoeffAlgo` per UE group. IRC only in the ≤ 8-antenna coefficient kernel | Auto (SNR threshold), forced, or **dynamic per 4-RB cluster** (two-cluster split of RBG powers for the noise floor, INR > −3 dB) |

## 6. Equalizer coefficients and MMSE equalization

| | cuPHY | FlexRAN |
|---|---|---|
| Form | G = Hᴴ R_ww⁻¹ H + I, C = G⁻¹ Hᴴ R_ww⁻¹ (normalized form) | A = HᴴH + σ²I, W = A⁻¹Hᴴ (or the IRC form) |
| Inversion | **LDLᴴ** of G, then one augmented solve for [Ree \| C]; only diag(Ree) is kept | **Explicit inverse** of A with the Cholesky-based `matrix_inv_cholesky_NxN` (`FR/lib_common/matrix.hpp:41-99`); `rcp14` for 1 × 1 |
| Bias | **Removed**: C scaled by λ = 1/(1 − Ree_ℓℓ), so z = x + e. λ ≤ 10 | **Kept**: x̂ = βx + ñ with β = diag(A⁻¹·HᴴH) per RE (`pMmseGain`). β ≤ 0.9999 (`m_gain_threshold`, `FR/lib_common/mimo.hpp:73`) |
| Granularity | Per subcarrier × time instance (≤ 4) | Per RE (subcarrier × channel symbol). The DFT-s-OFDM path averages β over the subcarriers of a symbol |

The two forms are mathematically equivalent. cuPHY's Ree = σ²(HᴴH + σ²I)⁻¹, so β = 1 − Ree_ℓℓ. The two clamps also match: FlexRAN's β ≤ 0.9999 limits SINR to 9999, and cuPHY's `MAX_ERROR_PRECISION` = 10000. Both stop at about 40 dB.

## 7. Demapping and LLRs

| | cuPHY | FlexRAN |
|---|---|---|
| Input | Unbiased z and ρ = (1 − Ree)/Ree | Biased x̂, β and 1/(1 − β) |
| Inner region | c = 4Aρ, LLR = 4Aρ·z = **4A·x̂/(1 − β)** | **4A·x̂/(1 − β)**, which is identical |
| Outer regions | **Simplified**: one slope per bit (16QAM: c·x and c(2A − \|x\|)) | **Exact max-log piecewise**: the slope steps up past each threshold (16QAM: 8/√10·(x̂ ± β/√10)/(1 − β)) (`FR/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr_avx512.h:134-146`) |
| Output | FP16, 8 slots per RE, plus the LLRCdm1 copy | int8 saturated (`nLlrFxpDecimal` fractional bits), in codeword order |

## 8. Measurements

| Measurement | cuPHY | FlexRAN |
|---|---|---|
| **Pre-eq SNR/SINR** | SinrPreEq = RSRP_dB − NoiseVarPreEq_dB. RSRP uses Σ\|Ĥ\|² with **no noise-bias subtraction**; NoiseVarPreEq includes +0.5 dB | Per (layer, antenna): S = mean\|β_DMRS·Ĥ\|² **− σ²·noiseFac** (`phy_cestimate_5gnr_simd.cpp:1730`), SNR = S/σ². Averaged over antennas, then over the UE's layers, minus 20log β_DMRS. EMA α = 0.125 in gnb |
| **Post-eq SINR** | Mean over REs and layers of ρ = (1 − Ree)/Ree at one time instance t*, per UE. Only `rsrpMeasKernel_v1` produces it | MIMO path: mean over REs of β/(1 − β), **per layer**, **first channel/data symbol only** (`FR/lib_common/mimo.hpp:349-351`; `phy_pusch_symbol_processing_5gnr_avx512.cpp:487, 821`). DFT-s path: β̄/(1 − β̄), with β averaged over subcarriers first, also first channel symbol only (`:1094-1103`) |
| **RSRP** | Σ over the UE's **layers** (summed, not averaged) of \|Ĥ\|² per RE per antenna, in dB of digital power | `nSigPwr[r]` is built from the "power bits": **mean\|Y_DMRS / β_DMRS\|²**, the received power (signal + interference + noise), **identical for every layer** (`phy_cestimate_5gnr_simd.cpp:2189, 2417-2426`; `calc_rx_pwr` in `phy_cestimate_5gnr_internal.hpp:886`). Converted to dBm with `Rx0DbfsRadioPower` |
| **RSSI** | Σ\|Y\|² over the allocation's subcarriers and **all antennas** on the DMRS symbols, divided by the number of symbols, per UE group, in dB. Also per symbol and antenna (`RssiFull`) ([pusch_rssi.cu:331-392](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L331-L392)) | (a) Wideband task: per RB, symbol and subband, summed over antennas, in dBm. (b) `nDMRSPwrDBFS`: in-allocation DMRS power averaged over antennas |
| **IoT per RBG** | — (only the per-PRB R_ww, used internally) | `nIntfNoisePwr[rbg]`, reported to MAC |

## 8.1 Numerical precision

**cuPHY uses floating point throughout. FlexRAN stores data in fixed point (int16, int8) and switches to float32 for the matrix math and the measurements.** On Sapphire Rapids (SPR), FlexRAN also has an FP16 path.

### Signal path

| Quantity | cuPHY | FlexRAN 24.03 |
|---|---|---|
| **Rx data Y** | **C16F** (`DataRx`), scaled by the fixed β_ul during decompression. C32F only in the `ASIM_CUPHY_FP32` build ([pusch_rx.cpp:1887-1890](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L1887-L1890)) | **int16 complex, Q13**, after the AGC shift (INT16 path). **FP16** in the SPR FP16 path |
| DMRS sequence | Generated inside the kernels | int16 **Q14** |
| Wiener filter weights | Static coefficient tensors (type not checked) | **int16** LUT; separate `_fp16.bin` tables for the FP16 path |
| LS estimate / **Ĥ** | **C32F** (`DmrsLSEst`, `HEst`; `cplxTypeCh = CUPHY_C_32F` is hard-coded, [pusch_rx.cpp:10715](../cuPHY/src/cuphy_channels/pusch_rx.cpp#L10715)) | **int16 complex, 16S13** (`pCeOut`). FP16 in the FP16 path |
| CE accumulation (noise, signal power) | FP32 | **float32** (INT16 path). In the FP16 path, **FP16** accumulators converted to float32 for the final sum |
| **Noise variance σ²** | **FP32**: `NoiseVarPreEq` in dB, `invNoiseVarLin` linear | **float32** (`nSigma2`; `fSigma2Mean` = σ²·2²⁶, i.e. Q26) |
| R_ww / R_nn | FP32 in shared memory. **L⁻¹ stored as C32F** (`LwInv`) | R_nn⁻¹ stored as **int32** real and imaginary planes per RBG (from the FlexRAN doc, not re-checked). The R_nn kernel also has an FP16 option |
| CFO / TA | C32F `CfoEst`; R32F `CfoHz` and `TaEst` | float32 (`fEstCfo`, `fEstAngleOffset`). TA as int16 samples or a TA index |
| **Equalizer A, A⁻¹ (or G, Ree)** | **FP32** (`TCompute = float`). **Exception:** the 64-antenna, 8-layer coefficient kernel computes in **FP16** (`TComputeHalf`, [channel_eq.cu:6535](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L6535)) | **float32** (`F32vec16`) in the INT16 path. **FP16** (`CF16vec16`, FP16 `matrix_inverse`) in the SPR FP16 path |
| **Eq coefficients C / W** | Stored as **C32F** (`EqCoef`) | Not stored: computed in registers and used at once in the fused kernel |
| Error precision / gain | **R32F** `ReeDiagInv`, capped at 10000 | β: float32, converted to **int16 16S15** for the demapper. 1/(1 − β): float × `llr_postsnr_fxp_dynamic`, then converted to int32. The FP16 path clamps it at 32759 |
| Equalized symbol x̂ | **C16F** (`DataEq`; debug or equalize-only). C32F for DFT-s-OFDM (`DataEqDft`) | **int16** (`pMmseOut`, NORMAL path). Not stored in the fused path |
| **LLR** | **FP16** (`R_16F`; R32F if built without `LLR_FP16`). Rate de-matching and the LDPC decoder input are also FP16 | **int8**, saturated to `nLlrSaturatedBits` = 6–8 bits, with `nLlrFxpDecimal` fractional bits (`FR/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr.h:110`) |

### Measurements: internal type and the L2 report

| Quantity | cuPHY internal | cuPHY → L2 (SCF adapter) | FlexRAN internal | FlexRAN → L2 |
|---|---|---|---|---|
| **Pre-eq SINR / SNR** | R32F, dB | `ul_sinr_metric`: int16, **×500 (0.002 dB)** ([scf_5g_fapi_phy.cpp:3715](../cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L3715)) | float32, linear | `nSNR` int16, **1/256 dB** |
| **Post-eq SINR** | R32F, dB | Used for `ul_cqi` = round(SINR + 64)·res when available ([:4298-4314](../cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4298-L4314)) | float32, linear | `nPostSNR` int16, 1/256 dB |
| Post-eq noise variance | R32F, dB | — | — | — |
| **RSRP** | R32F, dB | uint16 = **round(dBm + 140)·10**, clamped to [−140, −12] dBm ([:4270-4273](../cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4270-L4273)) | power bits, float32 | `nSigPwr` / `nTotalSigPwr` int16, 1/256 dBm |
| **RSSI** | R32F, dB; also R32F `RssiFull` per symbol and antenna | uint16 = **round(dBm + 128)·10**, clamped to [−128, 0] dBm ([:4228-4231](../cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp#L4228-L4231)) | float32, linear | `nRssi` int16, 1/256 dBm. `nDMRSPwrDBFS` in 0.01 dB |
| Interference + noise | Per-PRB R32F, internal only | Interference in dBm (`pn_dbm`) | float32 per RBG | `nIntfNoisePwr` int16, 1/256 dBm |
| TA | R32F, µs | `timing_advance_ns`: int16 = µs × 1000 | float32, samples | `nTA` index, `nTANanos` |

### Notable points

1. **Rx data and channel.**
   - cuPHY's FP16 input carries about 11 significant bits over a very wide dynamic range, so no AGC is needed.
   - FlexRAN's int16 Q13 input carries 16 bits but needs AGC to stay in range (§1.1).
   - cuPHY keeps Ĥ in **FP32**. FlexRAN quantizes Ĥ to **int16**.
2. **Matrix inversion.** Both stacks use float32, with two exceptions that invert in **FP16**: FlexRAN's SPR FP16 path and cuPHY's 64-antenna, 8-layer kernel.
3. **LLRs: the largest difference.**
   - cuPHY passes **FP16** LLRs to an FP16 LDPC decoder: wide range, no saturation.
   - FlexRAN passes **int8** LLRs, saturated to 6–8 bits, as the ACC100/ACC200 accelerator expects.
   - As a result, FlexRAN's LLR fixed-point scaling (`llr_postsnr_fxp_dynamic`, `nLlrFxpDecimal`) affects decoding performance. cuPHY has no equivalent.
4. **Report resolution: possible precision loss in cuPHY-CP.**
   - The FAPI fields for RSRP and RSSI are in 0.1 dB units, but the adapter computes `std::round(dBm + 140.0) * 10`. It rounds to whole dB before scaling by 10, so the reported RSRP and RSSI have an effective resolution of **1 dB**. `round((dBm + 140) * 10)` would keep 0.1 dB.
   - This is in the cuPHY-CP SCF adapter, not in cuPHY. It has not been changed.
   - FlexRAN reports with 1/256 dB resolution. The cuPHY SINR metric (0.002 dB) is not affected.

## 9. Key practical differences

- **Channel estimation.** FlexRAN adapts its filter to SNR and delay spread every slot. cuPHY uses fixed filters and adapts only the delay centring.
- **Noise.** FlexRAN corrects σ² for the noise the filter removed and passes the corrected value to the equalizer. cuPHY applies only a +0.5 dB correction, and only to the reported value.
- **Equalizer.** The math is the same; the bias is handled differently. cuPHY removes the bias in the coefficients. FlexRAN keeps it and absorbs it into the LLR scale. The LLRs are identical in the inner region. FlexRAN's exact max-log outer segments differ from cuPHY's simplified demapper.
- **Measurements.** The two "RSRP" values are **not comparable**: cuPHY reports |Ĥ|² summed over layers, while FlexRAN reports total received DMRS power per antenna. Post-equalization SINR agrees in definition: both are a linear mean of β/(1 − β) = ρ.

## 10. Corrections applied to the FlexRAN doc

Each item below was checked against the code and has been corrected in the FlexRAN doc.

1. The kernel source is in the tree, under `sdk/source/phy/lib_*`. This had already been corrected by a separate edit of the FlexRAN doc, which also added SDK links.
2. §3.4.3: W is a precomputed sinc LUT indexed by SNR in 0.25 dB steps. The default is 1 RB (`nInterpMode = 0`), not 4 RBs.
3. §3.4.4: α = 1/(1 − Σ noiseFac), applied only when μ ≤ 2 and ≤ 2 layers. The signal power has σ²·noiseFac subtracted.
4. §3.4.4 and §4.4: the power bits are the received DMRS power **divided by** β_DMRS, and the same for all layers. They do not measure desired-signal power.
5. §3.6: there is also an INR-adaptive shrinkage mode (`N_REFINE = 2`).
6. §3.4.6: the noise term in R₀ is 0.095·σ², not 0.1·σ², and |R₁|/R₀ is clipped to 1.
7. §3.9.3 and §4.3: fPostSINR uses only the first channel (or data) symbol in both paths. In the DFT-s path, β is also averaged before β/(1 − β) is formed. β is capped at 0.9999.
