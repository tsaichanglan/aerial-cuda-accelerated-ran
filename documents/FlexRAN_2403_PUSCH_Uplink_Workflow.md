# FlexRAN 24.03 gNB Uplink: The PUSCH Receiver Workflow

This document follows the PUSCH receive chain in `source/nr5g/gnb_ul`. It covers each processing block, the math it implements, and what goes in and out of it. It also covers the uplink measurements reported to L2: SNR, post-equalisation SINR, RSRP-like signal power, RSSI, interference-plus-noise power, timing advance, and frequency offset.

---

## 0. Scope and Sources

| Source | Used for |
|---|---|
| `gnb_ul/gnb_ul_phy_pusch.c` | PUSCH pipeline: DMRS generation, channel estimation (CE), IRC covariance, equalisation, phase noise, demapping, descrambling, TB CRC |
| `gnb_ul/gnb_ul_phy_ebbu_pool_tasks.c` | Task graph (eBBU pool) and the order in which the stages run |
| `gnb_ul/gnb_ul_phy_rimrs.c` | Wideband RSSI measurement task |
| `api/phy2mac/gnb_phy2mac_api_rxsdu_proc.c` | Conversion of measurements to dBm/dB and filling of `CRC.indication`, `RSSI.indication` |
| `gnb_common/gnb_phy_conn_users_info.c/.h` | Per-UE smoothing of SNR and Doppler, DTX thresholds, `UL_TA_ZERO` |
| `common/inc/common_ran_parameters.h` | Fixed-point constants |
| `sdk/source/phy/lib_*` (`D:\FlexRAN\v24_03\sdk`) | `bblib_*` kernels: AGC detection and CE ([`lib_cestimate_5gnr`](../../../sdk/source/phy/lib_cestimate_5gnr/)), IRC covariance ([`lib_irc_rnn_calculation_5gnr`](../../../sdk/source/phy/lib_irc_rnn_calculation_5gnr/)), fused equaliser ([`lib_pusch_symbol_processing_5gnr`](../../../sdk/source/phy/lib_pusch_symbol_processing_5gnr/), [`lib_pusch_irc_symbol_processing_5gnr`](../../../sdk/source/phy/lib_pusch_irc_symbol_processing_5gnr/)), LLR demapping ([`lib_llr_demapping`](../../../sdk/source/phy/lib_llr_demapping/)), and others |
| *FlexRAN L1 Algorithm Description v22.07* (doc 610016-12.0), Ch. 2 | Math inside the `bblib_*` SDK kernels |

> **Important.** The signal-processing kernels ([`bblib_cestimate_5gnr_*`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp), [`bblib_pusch_symbol_processing_*`](../../../sdk/source/phy/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr_avx512.cpp), [`bblib_irc_rnn_calculation_*`](../../../sdk/source/phy/lib_irc_rnn_calculation_5gnr/phy_irc_rnn_calculation_5gnr_avx512.cpp), [`bblib_llr_demapping_*`](../../../sdk/source/phy/lib_llr_demapping/phy_llr_demapping.cpp), …) belong to the FlexRAN SDK (`sdk/source/phy/`, i.e. `D:\FlexRAN\v24_03\sdk`). Their source is **not** under `source/nr5g`, but it is available in the SDK tree, and every kernel name in this document links to the SDK file that implements it. In this document:
> * **Orchestration, parameters, buffer layout, measurement post-processing and reporting** are taken directly from the gnb_ul / api source.
> * **Kernel-internal math** follows the Intel *L1 Algorithm Description*. The equations in that PDF could not be extracted verbatim (its math glyphs use private fonts). They are therefore written here in standard notation that matches the document's structure and the kernel interfaces. Where the code and the document differ, the difference is stated.
> * **Checked against the SDK source:** the Wiener filter (§3.4.3), noise and signal power and power bits (§3.4.4), Doppler (§3.4.6), R_nn refinement (§3.6), post-SINR (§3.9.3, §4.3) and the reported signal power (§4.4) were compared with the SDK code and corrected where they differed.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $k$ | subcarrier index inside the PUSCH allocation ($0 \dots 12N_{RB}-1$) |
| $l$ | OFDM symbol index in the slot ($0\dots13$) |
| $r$ | receive antenna (virtual UL port) index, $r = 0\dots N_R-1$ |
| $t$ | transmission layer / DMRS port index, $t = 0\dots N_L-1$ |
| $u$ | UE index within an MU-MIMO group |
| $Y(k,l,r)$ | received frequency-domain sample (after FFT, decompression and AGC), Q13 fixed point (`PHY_RX_DATA_FIXED_POINT = 13`) |
| $X_{RS}(k,l)$ | DMRS base sequence; $w_f(k')$ and $w_t(l')$ are the frequency and time OCC |
| $\hat H(k,l,r,t)$ | channel estimate |
| $\mathbf H(k,l)\in\mathbb C^{N_R\times N_L}$ | MIMO channel matrix on RE $(k,l)$ |
| $\sigma^2$ | noise variance; $\mathbf R_{nn}$ is the interference-plus-noise covariance |
| $N$ | FFT size (`nULFftSize`); $N_{CP}$ is the CP length; $f_s = N\Delta f$ |
| $\Delta f = 15\cdot 2^\mu$ kHz | subcarrier spacing |
| $\beta_{DMRS}$ | DMRS power boost (`fDmrsBeta`, linear amplitude) |

---

## 2. Top-Level Workflow

### 2.1 Two pipelines

The code selects one of two pipelines with `pConfigParams->nCompressPUSCHProcess` (set in `gnb_main/gnb_main_interface.c`):

| Mode | Numerology | Equaliser path |
|---|---|---|
| `PHY_PUSCH_COMPRESS` (fused) | $\mu \le 1$ (FR1) | A single kernel, [`bblib_pusch_symbol_processing_*`](../../../sdk/source/phy/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr_avx512.cpp), does FO compensation, time interpolation, MMSE/IRC equalisation, (PT-RS), layer demapping and LLR demapping in one pass. Optionally, MLD-TS runs on a bbdev accelerator. |
| `PHY_PUSCH_NORMAL` (staged) | $\mu = 3$ (FR2 / mmWave) | The stages run separately: MMSE/IRC → PT-RS phase noise → RE demap → layer demap / IDFT → LLR demap |

Every slot is processed as **two half-slots**. The `flag` argument is 0 for the "CE0 / Equalizer0" half and 1 for the "CE7 / Equalizer7" half. Each half covers its own DMRS symbols (`nNrOfDmrsSymbol[flag]`, `nDmrsSymbolStart[flag]`) and data symbols (`nNrOfDataSymbol[flag]`, `nDataSymbolIdx[flag][]`). Because of this split, equalisation of the first half can start before the second half has arrived over the fronthaul.

### 2.2 Task graph (eBBU pool)

```
 UL_CONFIG  (phy_ebbu_pool_task_ul_config)
   └─ phy_ul_pusch_symbol_buffer_pre_alloc_func()        buffer allocation
 PUSCH_DMRS (phy_ebbu_pool_task_ul_pusch_dmrs)
   └─ phy_ul_pusch_dmrs_func(flag=0,1)                   DMRS / ZC base sequences
 PUSCH_CE0 / PUSCH_CE7   (split by UE group or by Rx antenna)
   └─ phy_ul_pusch_ce_func(flag)                         AGC, LS, TO, MMSE-CE, sigma^2, SNR, CFO, Doppler, IpN
 PUSCH_RNN0 / PUSCH_RNN7 (split by RBG)
   └─ phy_ul_pusch_rnn_func(flag)                        IRC: R_nn per 4-RB RBG + inversion
 [pre-task] avg_pusch_ce(flag)                           combine CE splits: sigma^2, SNR, TA, CFO
 PUSCH_EQUALIZER0 / EQUALIZER7 (split by UE group or RB)
   ├─ COMPRESS: phy_ul_pusch_symbol_process_func(flag)   (MMSE|IRC) + FO comp + interp + LLR
   │            (MLD: phy_ul_pusch_pre_mldts_func + bbdev MLD-TS)
   └─ NORMAL:   phy_ul_pusch_mmse_func(flag)
                phy_ebbu_pool_task_ul_pusch_symb_proc(flag):
                  phy_ul_pusch_pn_func -> de_remap -> de_layermap -> de_mod
 [MLD only] PUSCH_MLD_POST
   └─ phy_ul_pusch_post_mldts_func                       MLD-TS LLR re-ordering per UE
 PUSCH_LLR_RX (split by UE)
   ├─ (NORMAL) phy_ul_pusch_de_mod_func(flag=1)
   ├─ phy_ul_pusch_de_scramble_func                      Gold-sequence descrambling
   └─ phy_ul_pusch_uci_decode_func                       UCI-on-PUSCH demux and decode
 PUSCH_FEC_SETUP / PUSCH_DECODE
   └─ gnb_io_fec_ul_bbdev_ldpc_decoder_run_func          rate de-matching + LDPC + HARQ combining (bbdev / ACC)
 PUSCH_TB
   └─ phy_ul_bbdev_pusch_tb_func                         TB CRC, throughput stats -> RX_DATA / CRC.indication
```

### 2.3 Function block summary

| # | Function (file:line) | SDK kernel(s) | Input | Output |
|---|---|---|---|---|
| 1 | [`phy_ul_pusch_symbol_buffer_pre_alloc_func`](../gnb_ul/gnb_ul_phy_pusch.c#L166) | – | Slot grant list (`sPuschGroupStruct[]`, `sPuschRxPara[]`) | Pool blocks for DMRS sequences, `pCeOut`, `pCEIpNOut`, `pIRCRnnOutRe/Im`, `pMmseOut`, `pPostSINR`, `pDeRemapDataOut`, `pDeLayermapOut`, `pDeScrambleOut`, MLD `pQHyOut`/`pROut` |
| 2 | [`phy_ul_pusch_dmrs_func`](../gnb_ul/gnb_ul_phy_pusch.c#L405) | [`bblib_dmrs_base_seq_gen_avx512`](../../../sdk/source/phy/lib_pusch_dmrs_gen_5gnr/phy_pusch_dmrs_gen_5gnr_avx512.cpp), [`bblib_zc_sequence_gen*`](../../../sdk/source/phy/lib_zc_sequence_gen/phy_zc_sequence_gen.cpp), [`bblib_prbs_basic`](../../../sdk/source/phy/lib_common/pseudo_random_seq_gen.cpp) | $N_{ID}^{n_{SCID}}$, $n_{SCID}$, slot, DMRS symbol indices, RB range, DMRS type, TP flag, hopping mode | `pDmrsBaseSeq[ue][sym]`: DMRS base sequence (Q14) |
| 3 | [`phy_ul_pusch_ce_func`](../gnb_ul/gnb_ul_phy_pusch.c#L876) | [`bblib_agc_detection`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr.cpp), [`bblib_cestimate_5gnr_avx512`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp) (MMSE CE) or [`bblib_cestimate_dct_5gnr`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_dct_5gnr_avx512.cpp) | DMRS REs `pCEin[port][sym]`, base sequences, DMRS ports, $\beta_{DMRS}$, CE algorithm flags | $\hat H$ → `pCeOut[layer][rx]`; $\sigma^2$ → `fSigma2Mean`; per-RBG IpN → `fNoiseIntfPwr`; residual → `pCEIpNOut`; TO → `nTaEst`; SNR → `fEstSnr`; power bits; CFO → `fEstAngleOffsetRx`, `fEstCfoRx`; Doppler → `fEstDopplerShift`; delay-spread index; AGC gain; IRC flag |
| 4 | [`avg_pusch_ce`](../gnb_ul/gnb_ul_phy_pusch.c#L98) | – | Per-split CE results | `fSigma2Mean[0]`, `fEstAvgSnr[t]`, `nTaAvg[t]`, `nTa[t]`, `fEstAngleOffset[t]`, `fEstCfo[t]` |
| 5 | [`phy_ul_pusch_rnn_func`](../gnb_ul/gnb_ul_phy_pusch.c#L1872) | [`bblib_irc_rnn_calculation_5gnr_avx512`](../../../sdk/source/phy/lib_irc_rnn_calculation_5gnr/phy_irc_rnn_calculation_5gnr_avx512.cpp), [`rnn_inverse_all`](../../../sdk/source/phy/lib_pusch_irc_symbol_processing_5gnr/phy_pusch_irc_symbol_processing_5gnr.cpp) | `pCEIpNOut` (DMRS residuals) | $\mathbf R_{nn}^{-1}$ per RBG and DMRS symbol → `pIRCRnnOutRe/Im` |
| 6a | [`phy_ul_pusch_symbol_process_func`](../gnb_ul/gnb_ul_phy_pusch.c#L2473) (COMPRESS) | [`bblib_pusch_symbol_processing_avx512`](../../../sdk/source/phy/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr_avx512.cpp) / [`bblib_pusch_irc_symbol_processing_avx512`](../../../sdk/source/phy/lib_pusch_irc_symbol_processing_5gnr/phy_pusch_irc_symbol_processing_5gnr_avx512.cpp) | Data REs `pMIMOin`, `pCeOut`, $\sigma^2$ or $\mathbf R_{nn}^{-1}$, CFO, interpolation granularity, PT-RS config, modulation, LLR format | LLRs (int8) in `pDeScrambleOut`; `pMmseGain`, `pPostSINR`; `fPostSINR[t]` |
| 6b | [`phy_ul_pusch_pre_mldts_func`](../gnb_ul/gnb_ul_phy_pusch.c#L2025) (MLD) | [`bblib_mldts_pre_process_5gnr_*`](../../../sdk/source/phy/lib_mldts_process_5gnr/phy_mldts_pre_process_5gnr_avx512.cpp) | $\hat H$, $y$, $\sigma^2$ | $\mathbf Q^H y$ (`pQHyOut`), $\mathbf R$ (`pROut`) for the bbdev MLD-TS engine |
| 6c | [`phy_ul_pusch_mmse_func`](../gnb_ul/gnb_ul_phy_pusch.c#L3429) (NORMAL) | [`bblib_mimo_mmse_detection`](../../../sdk/source/phy/lib_equalization/phy_rx_mimo_mmse_avx512.cpp), [`bblib_mmse_irc_mimo_5gnr`](../../../sdk/source/phy/lib_mmse_irc_mimo_5gnr/phy_mmse_irc_mimo_5gnr_avx512.cpp), [`bblib_pusch_focompensation_5gnr`](../../../sdk/source/phy/lib_pusch_focompensation_5gnr/phy_pusch_focompensation_5gnr_avx512.cpp), [`bblib_idft_burst_fxp`](../../../sdk/source/phy/lib_dft_idft/phy_dft_idft.cpp) | `pMIMOin`, `pCeOut`, $\sigma^2$ / $\mathbf R_{nn}$ | $\hat x$ → `pMmseOut[layer]`; `pPostSINR`/`pMmseGain`; TP+PT-RS: `pTfCOut` |
| 7 | [`phy_ul_pusch_pn_func`](../gnb_ul/gnb_ul_phy_pusch.c#L4076) | [`ptrs_phase_noise_estimation_pre_func`](../phy_modules/pn_compensation/phy_rx_pn_compen.h), [`bblib_ptrs_phase_noise_estimation_5gnr`](../../../sdk/source/phy/lib_phase_noise_5gnr/phase_noise_5gnr_avx512.cpp), [`bblib_ptrs_phase_noise_compensation_5gnr`](../../../sdk/source/phy/lib_phase_noise_5gnr/phase_noise_5gnr_avx512.cpp) | PT-RS REs, DMRS $\hat H$, PT-RS config | Phase-corrected `pMmseOut` / `pTfCOut` |
| 8 | [`phy_ul_pusch_de_remap_func`](../gnb_ul/gnb_ul_phy_pusch.c#L4440) | `sch_re_demap_per_ue(_tfp)` | `pMmseOut` / `pTfCOut` | Data-only REs `pDeRemapDataOut` (DMRS and PT-RS removed) |
| 9 | [`phy_ul_pusch_de_layermap_func`](../gnb_ul/gnb_ul_phy_pusch.c#L4606) | [`bblib_layerdemapping_5gnr`](../../../sdk/source/phy/lib_layerdemapping_5gnr/phy_layerdemapping_5gnr.cpp), [`bblib_idft_burst_fxp`](../../../sdk/source/phy/lib_dft_idft/phy_dft_idft.cpp) | Per-layer RE streams | Codeword symbol stream `pDeLayermapOut` (IDFT applied for TP) |
| 10 | [`phy_ul_pusch_de_mod_func`](../gnb_ul/gnb_ul_phy_pusch.c#L4739) | [`bblib_llr_demapping_nn_5gnr`](../../../sdk/source/phy/lib_llr_demapping/phy_llr_demapping_5gnr_nearest_neighbour.cpp) | Equalised symbols | int8 LLRs in `pDeScrambleOut` |
| 11 | [`phy_ul_pusch_post_mldts_func`](../gnb_ul/gnb_ul_phy_pusch.c#L4860) | [`bblib_mldts_post_process_5gnr`](../../../sdk/source/phy/lib_mldts_process_5gnr/phy_mldts_post_process_5gnr.cpp) | MLD-TS output `pMldtsOut` | Per-UE LLRs |
| 12 | [`phy_ul_pusch_de_scramble_func`](../gnb_ul/gnb_ul_phy_pusch.c#L4960) | [`bblib_descramble_5gnr`](../../../sdk/source/phy/lib_scramble_5gnr/phy_scramble_5gnr.cpp) | LLRs, $c_{init}$ | Descrambled LLRs (in place) |
| 13 | `phy_ul_pusch_uci_decode_func` (`gnb_ul_phy_pusch_uci_decode.c`) | polar / RM / small-block decoders | Descrambled LLRs | HARQ-ACK, CSI-1, CSI-2; UL-SCH LLRs |
| 14 | `gnb_io_fec_ul_bbdev_ldpc_decoder_run_func` | DPDK bbdev (Intel ACC100/ACC200 or SW) | UL-SCH LLRs per CB | Decoded TB bits (`pDecodedData`) |
| 15 | [`phy_ul_bbdev_pusch_tb_func`](../gnb_ul/gnb_ul_phy_pusch.c#L5051) | `gnb_io_fec_ul_bbdev_ldpc_decoder_crc` | TB bits | `nTbCrcOut`; triggers `RX_DATA.indication` and `CRC.indication` |
| M | [`phy_ul_rssi_measurement_task`](../gnb_ul/gnb_ul_phy_rimrs.c#L527) | [`bblib_rimrs_rssi_measurement_5gnr_avx512`](../../../sdk/source/phy/lib_rimrs_rx_5gnr/phy_rimrs_rx_5gnr_avx512.cpp) | Configured symbols and subbands | Linear power per RB → `RSSI.indication` |

---

## 3. Block-by-Block Description and Math

### 3.1 Buffer pre-allocation: `phy_ul_pusch_symbol_buffer_pre_alloc_func`

* **Input:** `psPHYStateRx->nPUSCHGrantNum`, the UE groups (`sPuschGroupStruct[]`), and per-UE parameters (`sPuschRxPara[]`).
* **Output:** pointers carved from the `L1_MEM_UL_PUSCH` pools:
  * `pDmrsBaseSeq[ue][dmrs]`: $4\cdot(N_{RB}+RB_{start})\cdot N^{DMRS}_{sc/RB}$ bytes.
  * `pCeOut[t][r]`: $(N_{CE,0}+N_{CE,1})\times \text{RUP16}(12N_{RB})$ complex int16.
  * `pCEIpNOut[r][dmrs]`: DMRS residual for IRC.
  * `pIRCRnnOutRe/Im[sym][r][r']`: one int32 per RBG.
  * `pMmseOut[t]`, `pPostSINR[t]` (alias `pMmseGain`), `pTfCOut[t]` (TP+PT-RS), `pDeRemapDataOut[t][flag]`, `pDeLayermapOut[flag]` (NORMAL path only).
  * `pDeScrambleOut`: $14\cdot N_L\cdot 12N_{RB}\cdot Q_m$ bytes (int8 LLRs).
  * MLD: `pQHyOut[sym]`, `pROut[sym]` of size $(N_L^2+N_L+1)\cdot 12N_{RB}$ int16.

### 3.2 DMRS sequence generation: `phy_ul_pusch_dmrs_func`

**Input:** `nNidnScid`, `nScid`, `nSlotInFrame`, `nDmrsSymbolIdx[]`, `nRbStart`, `nRbSize`, `nDmrsType`, `nTransPrecodeEnable`, `nTpPuschId`, `nDmrsGroupSeqHop`.
**Output:** `pDmrsBaseSeq[idx][sym]` in Q14 (`nDmrsScale = 14`). With MMSE-CE, identical sequences are shared between UEs; the dedup key is $2N_{ID}^{n_{SCID}}+n_{SCID}$.

**CP-OFDM (TS 38.211 §6.4.1.1.1.1):**

$$
c_{init} = \Big(2^{17}\big(N^{slot}_{symb}\,n_{s,f}^{\mu}+l+1\big)\big(2N_{ID}^{n_{SCID}}+1\big) + 2N_{ID}^{n_{SCID}} + n_{SCID}\Big) \bmod 2^{31}
$$

$$
r(n)=\tfrac{1}{\sqrt2}\big[(1-2c(2n)) + j\,(1-2c(2n+1))\big]
$$

The sequence is generated for $n\in[n_{start}, n_{start}+N_{RB}N^{DMRS}_{sc/RB})$ with $n_{start}=RB_{start}\cdot N^{DMRS}_{sc/RB}$. Here $N^{DMRS}_{sc/RB}=6$ for type 1 and $4$ for type 2. The RE mapping is $k = 4n+2k'+\Delta$ (type 1) or $k=6n+k'+\Delta$ (type 2), with OCC $w_f(k')\,w_t(l')$.

**Transform precoding (DFT-s-OFDM, §6.4.1.1.1.2):** the sequence is a low-PAPR Zadoff–Chu sequence $r^{(\alpha=0,\delta=1)}_{u,v}(n)$ of length $M_{ZC}=12N_{RB}/2$, with

$$
u = \big(f_{gh}+n_{ID}^{RS}\big)\bmod 30,\qquad
f_{gh}=\Big(\sum_{m=0}^{7}2^m c\big(8(N^{slot}_{symb}n_s+l)+m\big)\Big)\bmod 30
$$

* Group hopping uses $c_{init}=\lfloor n_{ID}/30\rfloor$. The code reads one PRBS byte, `bits[(14·ns + l)]`, and reduces it modulo 30.
* Sequence hopping uses $v=c(N^{slot}_{symb}n_s+l)$ when $M_{ZC}\ge 6N_{sc}^{RB}$; otherwise $v=0$.

### 3.3 PUSCH AGC (digital gain control, detected in `phy_ul_pusch_ce_func`, applied in CE and equaliser)

#### 3.3.1 Why AGC matters

The whole PUSCH chain runs on int16 samples (Q13) and on int16/int32 fixed-point accumulators. The received level changes a lot from slot to slot: a UE close to the radio unit and a UE at the cell edge differ by tens of dB. Without a per-allocation gain step, two things go wrong:

* **Overflow at strong signals.** Sums over antennas and subcarriers in the CE (LS, residual, power) and in the equaliser (H^H·H, H^H·y) saturate or wrap in int16/int32.
* **Loss of precision at weak signals.** Small samples use only a few LSBs. The estimated noise variance σ², the Wiener-filter output and the LLR scale all lose resolution. This is most visible in the noise and interference-plus-noise estimates, because the noise sits well below the signal.

AGC therefore moves the DMRS REs of the allocation to a fixed working level (`agcTarget`) with a **power-of-two shift**. A shift is exact and costs one instruction per vector, and the gain stays constant for the whole slot. The same gain must be used by every kernel that touches this UE group. Otherwise the channel estimate $\hat H$ (derived from DMRS) and the data samples $y$ would no longer be on the same scale, and the MMSE/IRC equaliser would produce a biased output.

#### 3.3.2 Detection (`bblib_agc_detection`, run once per UE group)

AGC detection runs only when **all** of the following hold (`gnb_ul_phy_pusch.c:978-995, 1221-1264`):

* the CE method is MMSE CE (`PHY_PUSCH_MMSE_CE`; the DCT CE has no AGC),
* $\mu \neq 3$ (FR1 uses AGC, FR2 does not),
* `agcTarget ≠ 0`,
* this is the first DMRS group of the slot (`nDmrsSymStart == 0`, so it runs in the CE0 half).

The CE7 half does not repeat the detection. It reuses the gain stored in `pPuschGroupStruct->nEstAgcGain`.

The SDK kernel ([`bblib_agc_detection_type1/type2`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp#L4635)) works on the raw, un-shifted samples of the **first DMRS symbol only** (`p_ce_in[r][0]`), on the DMRS REs of the layer's CDM group:

$$
A_r=\sqrt{\frac{1}{N_{DMRS,sc}}\sum_{k\in\text{DMRS REs}}\big|Y_{RS}(k,l_0,r)\big|^2},\qquad
A_{rms}=\max_{r}A_r\;\;(\texttt{f\_sig\_rms})
$$

* The power is a mean over the DMRS REs of one antenna, and the antennas are combined with the **maximum**, not the mean. The strongest antenna sets the gain, so it does not overflow.
* The per-antenna value is overwritten by each layer's call. It is not accumulated over layers.

The gain and the power-domain scale are then

$$
g_{AGC}=\operatorname{round}\!\Big(\log_2\frac{A_{rms}}{A_{target}}\Big)\quad(\texttt{nEstAgcGain, int16}),\qquad
s_{AGC}=2^{2g_{AGC}}\quad(\texttt{fAgcScale})
$$

* $g_{AGC}>0$ is an arithmetic **right** shift (strong signal, attenuate). $g_{AGC}<0$ is a **left** shift (weak signal, amplify). The shift is applied on load as `_mm512_srai_epi16` or `_mm512_slli_epi16`.
* $A_{target}$ is `pPhyCfgVars->agcTarget`. Its default is not set in this source tree; it comes from the PHY configuration (the Intel document gives $2^{13}$).
* With a CE_RX_SPLIT (split by receive antenna), each split detects its own value. The first split to finish publishes the gain with an atomic compare-exchange on `nAgcGainRxSplit`, which starts as `INVALID_AGC_GAIN`, so that all splits use one gain (`gnb_ul_phy_ebbu_pool_tasks.c:214`).
* The sample buffer in memory is **not** modified. Each kernel shifts on load.

#### 3.3.3 Where the gain is applied

| Stage | Where | What uses $g_{AGC}$ |
|---|---|---|
| Detection | `gnb_ul_phy_pusch.c:1225` | Reads raw $Y$, produces $g_{AGC}$ and $s_{AGC}$. Stored in `nEstAgcGain` (group) and `p_ce_req->n_agc_gain`. |
| LS channel estimate | [`phy_cestimate_5gnr_simd.cpp`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp#L141) (`dmrs_ce_ls`, `dmrs_mmse_ce_ls`) | $Y_{RS}\gg g_{AGC}$ before the multiplication by $X^{*}_{RS}$ |
| SNR estimation, σ² / residual estimation | same file (`dmrs_mmse_ce_snr_est_5gnr`, `dmrs_mmse_ce_sigma2_est_5gnr`) | The same shift on every re-read of $Y_{RS}$, so $\hat\sigma^2$, IpN and the residual (`pCEIpNOut`) are all in the shifted domain |
| Time-offset estimation | `dmrs_ce_ta_comp_5gnr` | Works on the already-shifted LS estimate |
| Power bits | `calc_rx_pwr_bit` in [`phy_cestimate_5gnr_internal.hpp`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_internal.hpp#L921) | $b=\log_2\sqrt{P}+g_{AGC}$ restores the absolute level inside the kernel |
| IRC covariance | `phy_ul_pusch_rnn_func` | No explicit shift. The input `pCEIpNOut` is already in the shifted domain. |
| Fused equaliser | [`phy_pusch_symbol_processing_5gnr_avx512.cpp`](../../../sdk/source/phy/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr_avx512.cpp#L300) (`tx_calc`) | **Data** samples $y\gg g_{AGC}$ on load (`nAgcGain`, set at `gnb_ul_phy_pusch.c:2670`). The overflow counter `nAgcOverflow` is available (`AGC_OVERFLOW_DET`, off by default). |
| Fused IRC equaliser | [`phy_pusch_irc_symbol_processing_5gnr_avx512.cpp`](../../../sdk/source/phy/lib_pusch_irc_symbol_processing_5gnr/phy_pusch_irc_symbol_processing_5gnr_avx512.cpp#L1032) | Same shift on the data samples |
| MLD-TS pre-processing | [`phy_mldts_pre_process_5gnr_internal.h`](../../../sdk/source/phy/lib_mldts_process_5gnr/phy_mldts_pre_process_5gnr_internal.h#L148) (`nAgcGain` set at `gnb_ul_phy_pusch.c:2206`) | Same shift on the data samples |

**Not covered by AGC:**

* The wideband RSSI task (`phy_ul_rssi_measurement_task`) has its own scaling (`RIMRS_AGC_TARGET_BITS`) and does not use `nEstAgcGain`.
* PRACH has its own AGC (`PRACH_AGC_TARGET_BITS`).
* The DMRS-symbol dBFS report (§4.5b) uses the raw $Y_{RS}$.

**SPR / FP16 pipeline (`PipelineSPR`).** The kernels work in half-precision floating point. Detection is not used for the shift. `nEstAgcGain` is `nPuschSprCvtScale` (a configured conversion scale), or `f_sig_rms` when `nPuschSprCvtScale == 0`. The kernels then multiply by $1/\texttt{n\_agc\_gain}$.

#### 3.3.4 Restoring absolute levels

The data and DMRS are processed in the shifted domain, so every output that is an *absolute* power must undo the gain. A ratio does not need it.

| Quantity | Effect of $g_{AGC}$ | How it is handled |
|---|---|---|
| SNR, post-equalisation SINR, TA, CFO, Doppler | none (ratios and phases) | no correction |
| `fSigma2Mean` (equaliser σ²) | stays in the shifted domain | used as is, because the data are shifted by the same $g_{AGC}$ |
| Signal power (`fPowerBits`, §4.4) | restored inside the kernel | $b=\log_2\sqrt{P}+g_{AGC}$ |
| Interference-plus-noise per RBG (`fNoiseIntfPwr`, §4.7) | restored in gnb_ul | $\times\, s_{AGC}=2^{2g_{AGC}}$ at `gnb_ul_phy_pusch.c:1353-1368` |
| Debug noise level (`dmrsNoiseREdBm`) | restored in gnb_ul | $\hat\sigma^2\cdot 2^{2g_{AGC}}\cdot2^{26}$ (`gnb_ul_phy_pusch.c:1757`) |

If AGC is switched off (`agcTarget = 0`, μ=3, or DCT CE), then $g_{AGC}=0$ and $s_{AGC}=1$, so the same formulas apply with no scaling.

### 3.4 Channel estimation: `phy_ul_pusch_ce_func` → [`bblib_cestimate_5gnr`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp)

**Request fields set by gnb_ul:** `n_dmrs_config_type`, `n_rxant`, `n_layer`, `n_start_prb`, `n_prb`, `n_mu`, `n_fft_size`, `n_fl_dmrs_symb` (1 or 2 for single or double DMRS), `n_dmrs_symb`, `n_dmrs_symb_idx[]`, `n_data_symb`, `n_dmrs_port[t]`, `f_boost_linear[t]` ($\beta_{DMRS}$), `n_enable_ta_est/comp`, `n_interp_method`, `n_delay_spread_index`, `n_priori_ta`, `n_enable_doppler_est`, `n_enable_fo_comp`, `n_agc_gain`, `pCeTaFftShiftScCp` (TA phase table).

**Response:** `p_ce_out[r][t][sym]`, `p_ce_ls`, `p_irc_IpN[r][sym]`, `f_est_sigma2[]`, `f_est_IpnPwr[rbg]`, `f_est_snr[][t][r]`, `f_power_bits[r][t][]`, `n_est_ta[t][]`, `cf_foe_corr[t]`, `f_est_doppler_shift[t]`, `n_est_delay_spread_idx`.

The kernel runs these steps in order (Intel doc §2.7–2.14).

#### 3.4.1 LS estimate and OCC de-spread

For CDM group ports $t_0,t_1$ with frequency OCC $[1,1]$ and $[1,-1]$ on adjacent DMRS REs $k_{2m}, k_{2m+1}$ of the same CDM group:

$$
\tilde H(k,l,r)=\frac{Y_{RS}(k,l,r)}{\beta_{DMRS}\,X_{RS}(k,l)} = \frac{Y_{RS}(k,l,r)\,X^{*}_{RS}(k,l)}{\beta_{DMRS}}\quad(|X_{RS}|=1)
$$

$$
\hat H_{LS}(k_{2m},l,r,t_0)=\tfrac12\big[\tilde H(k_{2m},l,r)+\tilde H(k_{2m+1},l,r)\big],\qquad
\hat H_{LS}(k_{2m},l,r,t_1)=\tfrac12\big[\tilde H(k_{2m},l,r)-\tilde H(k_{2m+1},l,r)\big]
$$

* For type 1, $k_{2m},k_{2m+1}$ are spaced 2 subcarriers apart. For type 2 they are adjacent.
* **Double-symbol DMRS** (ports 4–7 for type 1, 6–11 for type 2) adds a **time de-spread** between the two adjacent symbols: $\hat H_{t}=\tfrac12(\hat H^{(l)}_t \pm \hat H^{(l+1)}_t)$, with $+$ for $w_t=[1,1]$ and $-$ for $w_t=[1,-1]$.
* **Enhanced option (Intel doc §2.8):** a common TA pre-correction is applied first, $\tilde H'(k)=\tilde H(k)\,e^{j2\pi k\,TO_{avg}/N}$. The composite "sum" and "difference" channels are then interpolated 1:2 (type 1) or 1:3 (type 2) so that their subcarrier indices align before de-spreading. This removes the bias that frequency selectivity causes in the OCC de-spread.

#### 3.4.2 Timing-offset (TO) estimation and compensation (doc §2.12)

The phase rotation between DMRS REs that are $L$ subcarriers apart gives the delay. $L=4$ for type 1 (REs 1↔5, 3↔7) and $L=6$ for type 2 (REs 1↔7, 2↔8):

$$
C_t=\sum_{r}\sum_{m}\hat H_{LS}(m,r,t)\,\hat H_{LS}^{*}(m+2,r,t)
$$

Here $m$ is the DMRS RE index, so $m+2$ is $L$ subcarriers away.

$$
\hat\tau_t\,[\text{samples}]=\frac{N}{2\pi L}\,\angle C_t
$$

Compensation on the DMRS LS estimate, before the frequency filter:

$$
\hat H_{c}(k,r,t)=\hat H_{LS}(k,r,t)\,e^{\,j2\pi k\hat\tau_t/N}
$$

This "centres" the impulse response so that a symmetric Wiener/DCT filter can be used. After filtering, the delay is restored with $e^{-j2\pi k\hat\tau_t/N}$ (de-compensation). This happens before the residual and noise computation, so the estimate stays time-aligned with the received data. Because the MMSE CE has two steps, the de-compensation is done twice.

gnb_ul then post-processes the TO:

* It is clamped to $\pm N_{CP,max}$ (from [`bblib_get_sys_params`](../../../sdk/source/phy/lib_common/phy_tafo_table_gen.cpp)) so that later table look-ups stay in range.
* It is stored as `nTaEst[t][r]` and averaged over antennas in `avg_pusch_ce`.
* It is converted to a TA index (see §4.6).

#### 3.4.3 MMSE (Wiener) frequency-domain filtering (doc §2.7.3)

The filter is applied per RB (default) or per 4-RB group, and per layer:

$$
\hat{\mathbf h}_{MMSE}=\mathbf W\,\hat{\mathbf h}_{c},\qquad
\mathbf W=\mathbf R_{H\hat H}\Big(\mathbf R_{\hat H\hat H}+\tfrac{1}{\mathrm{SNR}}\mathbf I\Big)^{-1}
$$

**W is not computed at run time.** It is read from **precomputed tables** shipped as [`lib_cestimate_5gnr/ce_type{1,2}_w_{,4rb_}sinc{,_fp16}.bin`](../../../sdk/source/phy/lib_cestimate_5gnr/) (`gCeType1WienerCoeffSinc`, `gCeType1WienerCoeff4RbSinc`, …). The table index is ([phy_cestimate_5gnr_simd.cpp:1393-1409, 4222-4275](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp)):

| Index | Values |
|---|---|
| μ | numerology |
| `n_delay_spread_index` | delay-spread level |
| DMRS CDM grid Δ | type 1: 2, type 2: 3 |
| SNR | −10 … 40 dB in 0.25 dB steps (201 points): `nSNRIdx = ⌊(SNR_dB + 0.1249)·4⌋ + 40`, SNR clamped to [−10, 40] dB |

| `nInterpMode` | Table | W size (type 1) |
|---|---|---|
| 0 (**default**) | 1 RB + sinc | 6 × 12 |
| other (INT16 AVX-512 only) | 4 RB + sinc | 24 × 48 |

The tables were generated offline from the **sinc** correlation model:

$$
R(k,k';l,l')=\operatorname{sinc}\!\big(2\pi\,\tau_{rms}\,\Delta f\,(k-k')\big)\cdot\operatorname{sinc}\!\big(2\pi f_D T_{sym}(l-l')\big)
$$

The time factor equals 1 for frequency-only filtering. An alternative Bessel model is $R=\frac{1}{1+j2\pi\tau_{rms}\Delta f(k-k')}\,J_0\big(2\pi f_D T(l-l')\big)$.

* $\tau_{rms}$ is chosen by `n_delay_spread_index`. If `n_enable_delay_spread_est` is on (type 1 only), the index is updated every slot from `n_est_delay_spread_idx`.
* **Two-step filtering:**
  1. Filter with the 1-RB table at a default SNR, $\mathrm{SNR}_0 = 30$ dB (`phy_cestimate_5gnr_simd.cpp:1387`).
  2. Estimate the noise and SNR (§3.4.4).
  3. Filter again, using the table entry for the estimated SNR (`:4204-4250`).
* The same weights $\mathbf W$ interpolate from DMRS REs to **all** 12 subcarriers per RB. The result, `pCeOut[t][r]`, holds one complex value per RE for each CE symbol (`nNrOfCeSymbol[flag]`). Q-format: int16 complex.

**DCT alternative** (`nCeMethod ≠ PHY_PUSCH_MMSE_CE`, [`bblib_cestimate_dct_5gnr`](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_dct_5gnr_avx512.cpp)). This path has no Wiener filter and no AGC. It denoises the LS estimate in the DCT domain with a hard threshold, then interpolates linearly. The code is shown for DMRS type 1. The type 2 function has the same structure.

Per DMRS symbol group, layer and group of four receive antennas, the steps are:

1. **LS** on the DMRS REs: $\hat H_{LS}=Y_{RS}\,X^{*}_{RS}/\beta_{DMRS}$. For the multi-layer case (below) the LS estimates of all DMRS symbols are averaged.
2. **Timing offset** (§3.4.2), only in the normal case. The multi-layer case does not estimate it (`nTaEnable = 0`).
3. **DCT:** $h_{DCT}=\mathrm{DCT}(\hat H_{LS})$. The input is zero-padded to a DCT size of 384, 480, 1152 or 1728 points. Supported DMRS RE counts are $N_{DMRS,sc}\in\{384,480,1056,1068,1632,1638\}$. Any other value stops the process. The number of receive antennas must be a multiple of 4, because four antennas are processed in parallel.
4. **Noise cancellation (hard threshold).** Let $N_{win}$ be the delay window and $N_{len}=\lfloor N_{DMRS,sc}/N_{L,grp}\rfloor$ the number of useful taps ($N_{L,grp}$ is `n_layer_in_group`). The taps $n\in[N_{win},N_{len})$ are *outside* the window and are treated as noise only:

$$
\bar P_{out}=\frac{1}{N_{len}-N_{win}}\sum_{n=N_{win}}^{N_{len}-1}|h_{DCT}(n)|^2,\qquad
\gamma_{th}=\kappa\,\bar P_{out}
$$

$$
\tilde h(n)=\begin{cases}h_{DCT}(n), & n<N_{win}\ \text{and}\ |h_{DCT}(n)|^2\ge\gamma_{th}\\ 0,&\text{otherwise}\end{cases}
$$

   The taps from $N_{win}$ up to the DCT size are zeroed before the IDCT (`memset(pIdctIn + nWindow…)`).

5. **IDCT:** $\hat H_{clean}=\mathrm{IDCT}(\tilde h)$.
6. **Linear frequency interpolation.** $\hat H_{clean}$ exists only on the DMRS REs. It is expanded to every subcarrier by averaging neighbours. For type 1: $\hat H(2n+1)=\tfrac12\big[\hat H(2n)+\hat H(2n-2)\big]$. The last subcarrier is extrapolated from the previous ones. The TA table is then applied if a timing offset was estimated. This step is **not** Wiener filtering. It is `dmrs_type1_dct_ce_freq_interp_linear_5gnr` (type 2: `dmrs_type2_..._linear_5gnr`).

The window length and threshold factor depend on the number of layers in the group:

| Case | $N_{win}$ | $\kappa$ | DMRS symbols |
|---|---|---|---|
| $N_{L,grp}\le 4\cdot N_{fl}$ (type 1; $N_{fl}$ = front-loaded DMRS symbols) | $\big\lfloor N_{CP,min}\,N_{DMRS,sc}/N\big\rfloor$ | 3.0 (`THRESHOLD_FACTOR_1`) | one estimate per DMRS symbol group |
| $N_{L,grp}> 4\cdot N_{fl}$ (multi-layer) | $\big\lfloor N_{CP,min}\,N_{DMRS,sc}/(2N)\big\rfloor$ | 5.0 (`THRESHOLD_FACTOR_2`) | LS averaged over all DMRS symbols, one group |

Here $N_{CP,min}$ comes from `bblib_get_sys_params`, and only a single front-loaded DMRS symbol is supported.

**Noise and signal power.** The noise power is the energy of the outside-window taps plus the energy of the in-window taps that fell below $\gamma_{th}$. The signal power is the energy of the in-window taps that were kept:

$$
\hat P_{n}=\frac{1}{N_{len}}\Big[\sum_{n\ge N_{win}}|h_{DCT}(n)|^2+\sum_{n<N_{win},\,|h_{DCT}(n)|^2<\gamma_{th}}|h_{DCT}(n)|^2\Big]\!\Big/ s^2,\qquad
\hat P_{s}=\frac{1}{N_{len}}\sum_{n<N_{win},\,|h_{DCT}(n)|^2\ge\gamma_{th}}|h_{DCT}(n)|^2\Big/ s^2
$$

where $s$ is the DCT output scaling. The SNR is $\hat P_s/\hat P_n$ per layer and antenna. It is `SNR_FOR_ZERO_INPUT` (0) when both powers are zero, and `SNR_FOR_ZERO_NOISE` (999999) when only the noise power is zero.

**Noise variance** (`dmrs_dct_ce_snr_calc`):

* Normal case: the noise powers of all layers are added per antenna, then averaged over antennas: $\hat\sigma^2=\frac1{N_R}\sum_r\sum_t\hat P_n(t,r)$.
* Multi-layer case ($N_{L,grp}> 4\cdot N_{fl}$): $\hat\sigma^2=\frac{1}{8\,N_L N_R}\sum_{t,r}\hat P_n(t,r)$. The extra factor 1/8 is fixed in the code.

#### 3.4.4 Noise variance, SNR and signal power

The DMRS **residual** per RE (stored in `pCEIpNOut` for IRC):

$$
e(k,l,r)=Y_{RS}(k,l,r)-\sum_{t}\beta_{DMRS}\,w_f(k')w_t(l')X_{RS}(k,l)\,\hat H(k,l,r,t)
$$

**Noise variance:**

$$
\hat\sigma^2=\alpha\cdot\operatorname*{mean}_{k\in\text{DMRS},\,r}\big|e(k,l,r)\big|^2
$$

$\alpha$ corrects for the part of the noise that the Wiener filter removed from $\hat H$, and therefore from the residual. It comes from a look-up table ([phy_cestimate_5gnr_simd.cpp:1661-1686, 4151-4197](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp)):

$$
\alpha=\frac{1}{1-\sum_{t\in\text{CDM group}}\texttt{noiseFac}_t},\qquad
\texttt{noiseFac}=\texttt{pnoiseFac\_dmrsType\{1,2\}}[\text{delay spread}][\text{layers}][\text{SCS}][\text{SNR bucket}]
$$

* In the first pass the SNR bucket is fixed at 30 dB. After that it is one of 11 buckets in 5 dB steps.
* The correction applies only when `noiseScaling == 1`, $\mu\le2$ and the CDM group has at most 2 layers (to avoid a negative factor). Otherwise $\alpha = 1$.
* **In the shipped L1 the correction is disabled.** `gnb_ul_phy_pusch.c` hard-codes `p_ce_req->noiseScaling = 0` ([gnb_ul_phy_pusch.c:1163](../gnb_ul/gnb_ul_phy_pusch.c)), and no other call site sets it. In practice, then, $\alpha = 1$ and every $\texttt{noiseFac}_t = 0$, so $\hat\sigma^2$ is the plain mean residual power. The LUT path exists in the kernel but only runs if a caller sets `noiseScaling = 1`.

**Per-RBG interference-plus-noise power** (`f_est_IpnPwr[rbg]`): the same mean, taken per 4-RB RBG.

**Signal power and SNR**, per layer and antenna. The noise that leaks into $|\hat H|^2$ is subtracted (`:1687-1733`, `:4324`):

$$
P_{S}(t,r)=\operatorname*{mean}_{k}\big|\beta_{DMRS}\hat H(k,l,r,t)\big|^2-\hat\sigma^2\cdot\texttt{noiseFac}_t,\qquad
\mathrm{SNR}(t,r)=\frac{P_S(t,r)}{\hat\sigma^2}\quad(\texttt{f\_est\_snr, linear})
$$

With `noiseScaling = 0`, as the shipped L1 sets it, $\texttt{noiseFac}_t = 0$. The subtraction term then vanishes and $P_S$ is just $\operatorname{mean}|\beta_{DMRS}\hat H|^2$.

**Power bits** (`f_power_bits`) are **not** derived from $P_S$. They are the **received DMRS power** (signal + interference + noise), divided by the DMRS boost, with the AGC gain restored (`calc_rx_pwr(xmmY, xmmBoostRcp, …)` at `:2189` and `calc_rx_pwr_bit` at `:2417`; [phy_cestimate_5gnr_internal.hpp:886-940](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_internal.hpp)):

$$
P_{rx}(r)=\operatorname*{mean}_{k\in\text{DMRS}}\Big|\frac{Y_{RS}(k,l,r)}{\beta_{DMRS}}\Big|^2,\qquad
b(r)=\tfrac12\log_2 P_{rx}(r),\qquad P_{rx}=2^{2b}
$$

The same $b(r)$ is written for **every layer** $t$ of the CDM group (`:2420-2427`).

In MMSE-CE mode the kernel averages the SNR over antennas, so gnb_ul copies the value from antenna 0 to all antennas.

**Fixed-point scaling in gnb_ul:**

$$
\texttt{fSigma2Mean}[split] = \hat\sigma^2 \cdot 2^{\texttt{MMSE\_NOISE\_POWER}},\qquad \texttt{MMSE\_NOISE\_POWER}=2\times13=26
$$

$$
\texttt{fNoiseIntfPwr}[rbg] = \texttt{f\_est\_IpnPwr}\cdot s_{AGC}
$$

If several UEs share an RBG, `fNoiseIntfPwr` is the average of their values.

#### 3.4.5 Frequency-offset (CFO) estimation with ≥2 DMRS symbols (doc §2.13)

This requires `n_enable_doppler_est`. The kernel returns the inter-DMRS correlation per layer:

$$
\texttt{cf\_foe\_corr}[t]=\sum_{r}\sum_{k}\hat H(k,l_1,r,t)\,\hat H^{*}(k,l_0,r,t)
$$

Here $l_0$ is the front-loaded DMRS and $l_1$ the first additional DMRS (`n_dmrs_symb_idx[n_fl_dmrs_symb]`). gnb_ul then sums the correlation over the layers of each UE and converts it ([gnb_ul_phy_pusch.c:1429-1462](../gnb_ul/gnb_ul_phy_pusch.c#L1429-L1462)):

$$
C_u=\sum_{t\in\mathcal T_u}\texttt{cf\_foe\_corr}[t],\qquad
\hat\phi_u=\frac{\operatorname{atan2}(\Im C_u,\Re C_u)}{(l_1-l_0)+\delta_{CP}}\quad[\text{rad / OFDM symbol}]\;(\texttt{fEstAngleOffset})
$$

$$
\hat f_{CFO,u}=\frac{\hat\phi_u\, f_s}{2\pi\,(N+N_{CP,min})}=\frac{\hat\phi_u}{2\pi T_{sym}}\quad[\text{Hz}]\;(\texttt{fEstCfo})
$$

* For $\mu=0$, when $l_0<6<l_1$ (the long CP of symbol 7 lies between the two DMRS), the code adds $\delta_{CP}=(N_{CP,max}-N_{CP,min})/(N+N_{CP,min})$.
  * *Note:* as written, this expression is evaluated in **integer** arithmetic (`int16 / (uint32 + int16)`), so $\delta_{CP}$ is effectively 0.
* The unambiguous range is $|\hat\phi|(l_1-l_0)<\pi$. For example, with $l_1-l_0=9$ at 30 kHz SCS, this is about $\pm1.6$ kHz.
* `avg_pusch_ce` averages `fEstAngleOffsetRx` and `fEstCfoRx` over antennas.
* The Intel document also describes a one-tap IIR smoothing (weights 0.5, 0.25, 0.1 for QPSK, 16QAM, 64QAM) and MRC weighting when there are more than 2 DMRS symbols. In the gnb_ul code only the first DMRS pair is used, and no IIR smoothing is applied at this level.

#### 3.4.6 Doppler spread estimation (doc §2.14)

$$
R_0=\Big|\frac{1}{2KR}\sum_{l\in\{l_0,l_1\}}\sum_{r}\sum_{k}|\hat H(k,l,r,t)|^2-0.095\,\hat\sigma^2\Big|,\qquad
R_1=\frac{1}{KR}\sum_{r}\sum_{k}\hat H^{*}(k,l_0,r,t)\hat H(k,l_1,r,t)
$$

$$
\hat f_D=\frac{1}{2\pi T}\sqrt{\frac{1-|R_1|/R_0}{0.223}}
$$

$T$ is the time between the two DMRS symbols. The constant 0.223 comes from a fitted approximation of $J_0(2\pi f_D T)$.

In the code ([phy_cestimate_5gnr_simd.cpp:2793-2814](../../../sdk/source/phy/lib_cestimate_5gnr/phy_cestimate_5gnr_simd.cpp)):
* The noise term is $0.095\,\hat\sigma^2$ per RE, not $0.1\,\hat\sigma^2$.
* If $|R_1/R_0| > 1$, the ratio is set to 1.
* The ratio is then averaged over antennas.

gnb_ul smooths the per-layer estimate with a scalar Kalman filter (`nr5g_gnb_conn_users_ul_doppler_calc`, $q=r=0.1$):

$$
K_n=\frac{P_{n-1}+q}{P_{n-1}+q+r},\quad
\bar f_{D,n}=\bar f_{D,n-1}+K_n(\hat f_{D,n}-\bar f_{D,n-1}),\quad
P_n=(1-K_n)(\bar f_{D,n}+q)
$$

The code's $P_n$ update uses the filtered Doppler value. The result, `fEstDopplerShift[t]`, drives the selection of the time-interpolation method (§3.7).

#### 3.4.7 IRC enable decision (end of CE)

| Condition | Effect |
|---|---|
| `nEnableIrc == 0`, $10\log_{10}\mathrm{SNR}\le$ `IrcEnableThreshold`, resource allocation type 1, $N_R\in\{2,4,8,16\}$ | `nIrcFlag = 1` (interference-limited, so IRC is used) |
| `nEnableIrc == 2` or `3`, allocation type 1, $N_R\in\{2,4,8,16\}$ | IRC forced on |
| `IrcDynamicFlag == 1` | Per-4-RB-cluster MMSE/IRC selection (below) |

**Dynamic IRC.** Let $p_i$ be the RBG IpN powers (`fNoiseIntfPwr`).

1. Sort the $p_i$ in ascending order.
2. Run a 1-D two-cluster split (at most 15 clusters, at most 9 iterations). The "low" cluster mean is the noise floor $\hat N_0$. If there is only one RBG, $\hat N_0=p_0/10$.
3. For each RBG compute

$$
\mathrm{INR}_i=\frac{p_i}{\hat N_0}-1,\qquad
\texttt{IRCFlagBuffer}[i]=\begin{cases}1 & \mathrm{INR}_i>10^{-0.3}\ (\approx -3\text{ dB})\\ 0&\text{otherwise}\end{cases}
$$

### 3.5 Combining CE splits: `avg_pusch_ce`

Called before the equaliser for each half-slot. With $S$ = number of CE splits that report a value:

$$
\bar\sigma^2=\frac1S\sum_{s}\texttt{fSigma2Mean}[s],\qquad
\overline{\mathrm{SNR}}_t=\frac1{N_R}\sum_r \mathrm{SNR}(t,r)\;(\texttt{fEstAvgSnr})
$$

$$
\bar\tau_t=\frac1{N_R}\sum_r\hat\tau(t,r),\quad
\texttt{nTa}[t]=\Big\lfloor \bar\tau_t\cdot\frac{128}{N}\Big\rfloor+31,\quad
\bar\phi_t=\frac1{N_R}\sum_r\hat\phi(t,r),\quad
\bar f_{CFO,t}=\frac1{N_R}\sum_r \hat f_{CFO}(t,r)
$$

### 3.6 IRC interference covariance: `phy_ul_pusch_rnn_func`

This runs when any group has `nIrcFlag == 1`, or when `IrcDynamicFlag == 1`. It is split by RBG. The input is the residual $e(k,l,r)$ from `pCEIpNOut`.

$$
\mathbf e(k,l)=[e(k,l,0),\dots,e(k,l,N_R-1)]^T,\qquad
\hat{\mathbf R}_{nn}(g,l)=\frac{1}{n}\sum_{k\in g}\mathbf e(k,l)\mathbf e^H(k,l),\quad n = 4\,\text{RB}\times N^{DMRS}_{sc/RB}
$$

* **Refinement (shrinkage)** of $\hat{\mathbf R}_{nn}$ toward a scaled identity keeps the matrix well conditioned ([phy_irc_rnn_calculation_5gnr_avx512.cpp:45-85, 256-273](../../../sdk/source/phy/lib_irc_rnn_calculation_5gnr/phy_irc_rnn_calculation_5gnr_avx512.cpp)):

  $$
  \hat{\mathbf R}_{nn}\leftarrow(1-\alpha)\,\hat{\mathbf R}_{nn}+\alpha\,\frac{\operatorname{tr}\hat{\mathbf R}_{nn}}{N_R}\,\mathbf I
  $$

  | Mode (`N_REFINE`) | α |
  |---|---|
  | 1 (`RNN_REFINE`) | Fixed 0.25 (`ALPHA_FACTOR`; `BETA_FACTOR` = 0.75) |
  | 2 | Adaptive: $\alpha = a\cdot\mathrm{INR_{dB}} + b$ from `shrinkage_factor_table[nRx ∈ {2,4,8,≥16}][INR bucket ∈ {≤0, 0–3, 3–6, 6–9, >9 dB}]`. Applied only to RBGs with `pIrcFlag` set |
* [`rnn_inverse_all`](../../../sdk/source/phy/lib_pusch_irc_symbol_processing_5gnr/phy_pusch_irc_symbol_processing_5gnr.cpp) stores $\hat{\mathbf R}_{nn}^{-1}$ per RBG and DMRS symbol (int32 real and imaginary planes).
* With `avg2sym = 1` (linear interpolation and 2 CE symbols), the two DMRS symbols are averaged.
* With `nEnableIrc == 2` (2-SC processing) or `3`, the inversion is left to the symbol-processing kernel (`nDisableRnnInv = 1`).

### 3.7 Time-domain interpolation of the channel (doc §2.10 and §2.15)

The CE produces $\hat H$ on the DMRS (CE) symbols only. For each data symbol $l$:

* **Nearest** (default when `nLinInterpEnable = 0`): $\hat H(l)=\hat H(l_{j^{*}})$ with $j^{*}=\arg\min_j|l-l_j|$. gnb_ul pre-computes `nSymbPerDmrs[j]`, the number of data symbols mapped to each DMRS.
* **Linear (1+1):**

  $$\hat H(l)=\frac{l_1-l}{l_1-l_0}\hat H(l_0)+\frac{l-l_0}{l_1-l_0}\hat H(l_1)$$

* **Lagrange (1+1+1):**

  $$\hat H(l)=\sum_{j=0}^{2}\hat H(l_j)\prod_{i\neq j}\frac{l-l_i}{l_j-l_i}$$

* **Grouped linear ("Linear-G", G ∈ {1, 2, 3, 4, 6}):** consecutive symbols share one interpolated channel evaluated at their mean index. This reduces the number of distinct equaliser matrices. The number of groups is the `nMldRSymbNum` table: {12, 6, 4, 3, 2} for 2 DMRS and {10, 5, 4, 3, 2} for 4 DMRS.

**Granularity selection** (`nLinearInterpGranularity == BBLIB_INTERP_GRANS`, FO or Doppler estimation enabled):

* `interpTable = 0`: `gPuschLinInterGranTable[cfgId][Qm-index][doppler-index]` with a Doppler step of $500/\texttt{DOPPLER\_SIZE}$ Hz. The minimum over layers is used.
* `interpTable = 1` or `2` (mapping type A with DMRS patterns {2,11}, {2,7,11} or {2,5,8,11}): compute $\nu=f_D/\Delta f$ and quantise SNR and delay spread. For each method $m$ with tabulated normalised MSE $\epsilon_m$:

  $$
  D_m = 1+\sum_{t}\epsilon_m\,\overline{\mathrm{SNR}}_t,\qquad
  m^{*}=\max\{m: 10\log_{10}D_m-10\log_{10}D_0<0.2\text{ dB}\}
  $$

  $m^{*}=6$ means nearest. Linear-5 is not supported (it maps to 3). With table 2 the method also comes with tabulated interpolation weights (`pIntrpweights`).

### 3.8 Frequency-offset compensation in the equaliser (doc §2.13.2)

The data cannot be de-rotated before MIMO combining, because the REs contain several UEs. Instead, the **channel** is rotated:

$$
\hat H'(k,l_j,r,t)=\hat H(k,l_j,r,t)\,e^{-j\hat\phi_t\,l_j}\;\xrightarrow{\text{time interp}}\;\hat H'(k,l,r,t)\;\rightarrow\;
\hat H(k,l,r,t)=\hat H'(k,l,r,t)\,e^{\,j\hat\phi_t\,l}
$$

* COMPRESS path: `fEstCfo[t] = fEstAngleOffset[t]` (rad/symbol), with the CP-aware table `pFoCompScCp`. The exact per-symbol phase is $2\pi f_{CFO}\,(l(N+N_{CP})+N_{CP})/f_s$.
* NORMAL path: `mmse_mimo_request.fEstCfo` is given `fEstCfo[t]` in Hz.
* Also in the NORMAL path, [`bblib_pusch_focompensation_5gnr`](../../../sdk/source/phy/lib_pusch_focompensation_5gnr/phy_pusch_focompensation_5gnr_avx512.cpp) can de-rotate the raw data in place using `f_offset_angel_est`. That value is currently forced to 0 ("mmWave FOC need to be calibrated").

### 3.9 MIMO equalisation (doc §2.16)

The received data model per data RE:

$$
\mathbf y(k,l)=\mathbf H(k,l)\,\mathbf x(k,l)+\mathbf n(k,l),\qquad \mathbf y\in\mathbb C^{N_R},\ \mathbf x\in\mathbb C^{N_L}
$$

**Input:** `pRxSignal[r][sym]` (data REs), `pChState[r][t]`, `nSigma2`, `nAgcGain`, interpolation and FO settings.
**Output:** LLRs (COMPRESS) or $\hat{\mathbf x}$ (NORMAL), plus `pMmseGain` / `pPostSINR` and `fPostSINR[t]`.

#### 3.9.1 MMSE (default, `nIrcFlag = 0`)

$$
\mathbf W_{MMSE}=\big(\mathbf H^H\mathbf H+\sigma^2\mathbf I_{N_L}\big)^{-1}\mathbf H^H,\qquad
\hat{\mathbf x}=\mathbf W_{MMSE}\,\mathbf y
$$

with

$$
\sigma^2 = \texttt{fSigma2Mean}[0]\;\big(\times\texttt{PuschNoiseScale}\ \text{if } N_L^{UE}>1,\ \text{single UE},\ \mu=0\big)
$$

The extra scaling for multi-layer single-UE at $\mu=0$ makes the equaliser more robust to CE error. Because $Y$ is Q13, $|Y|^2$ is Q26, which is why $\sigma^2$ carries the factor $2^{26}$.

#### 3.9.2 MMSE-IRC (`nIrcFlag = 1`, or per RBG in dynamic mode)

$$
\mathbf W_{IRC}=\mathbf H^H\big(\mathbf H\mathbf H^H+\hat{\mathbf R}_{nn}\big)^{-1}
=\big(\mathbf H^H\hat{\mathbf R}_{nn}^{-1}\mathbf H+\mathbf I\big)^{-1}\mathbf H^H\hat{\mathbf R}_{nn}^{-1}
$$

$\hat{\mathbf R}_{nn}^{-1}$ is taken from the RBG that contains $k$ and from the nearest DMRS symbol. In dynamic mode the allocation is processed in 4-RB chunks: chunks with `IRCFlagBuffer = 1` use IRC and the others use MMSE.

#### 3.9.3 Equalisation gain and post-equalisation SINR

The MMSE estimate is biased:

$$
\hat x_t=\beta_t\,x_t+\tilde n_t,\qquad
\beta_t=[\mathbf W\mathbf H]_{tt}=1-\sigma^2\big[(\mathbf H^H\mathbf H+\sigma^2\mathbf I)^{-1}\big]_{tt}
$$

$$
\boxed{\ \mathrm{SINR}^{post}_t=\frac{\beta_t}{1-\beta_t}=\frac{1}{\sigma^2\big[(\mathbf H^H\mathbf H+\sigma^2\mathbf I)^{-1}\big]_{tt}}-1\ }
$$

$\beta_t$ is stored per RE (`pMmseGain`). It is clamped to $\beta_t\le0.9999$ (`m_gain_threshold`, [lib_common/mimo.hpp:73](../../../sdk/source/phy/lib_common/mimo.hpp)), which limits $\mathrm{SINR}^{post}$ to 9999 (about 40 dB). The LLR scale $1/(1-\beta_t)$ uses the clamped value (`mimo.hpp:334-357`). How `fPostSINR[t]` is formed depends on the path:

| Path | `fPostSINR[t]` | Source |
|---|---|---|
| MIMO (CP-OFDM) | $\frac{1}{N_{sc}}\sum_k \frac{\beta_t(k)}{1-\beta_t(k)}$: the mean of the per-RE SINR, **over the first channel symbol only**. Without interpolation that is `iChSymb == 0` (`:487`); with interpolation it is the first data symbol, `iSymb == 0` (`:821`) | `mimo.hpp:349-351`; [phy_pusch_symbol_processing_5gnr_avx512.cpp:573](../../../sdk/source/phy/lib_pusch_symbol_processing_5gnr/phy_pusch_symbol_processing_5gnr_avx512.cpp) |
| DFT-s-OFDM (1 layer) | $\bar\beta/(1-\bar\beta)$ with $\bar\beta=\frac1{N_{sc}}\sum_k\beta(k)$, i.e. **β is averaged first**. Only the **first channel symbol** is used (`iChSymb == 0`). In this path the LLR scale also uses the per-symbol $\bar\beta$ instead of per-RE values | `phy_pusch_symbol_processing_5gnr_avx512.cpp:1094-1120` |

The value is reported as `nPostSNR` (§4.3). For IRC, replace $\sigma^2\mathbf I$ with $\hat{\mathbf R}_{nn}$ and $\mathbf H^H\mathbf H$ with $\mathbf H^H\hat{\mathbf R}_{nn}^{-1}\mathbf H$, which gives $\beta_t=[(\mathbf G+\mathbf I)^{-1}\mathbf G]_{tt}$ with $\mathbf G=\mathbf H^H\hat{\mathbf R}_{nn}^{-1}\mathbf H$.

#### 3.9.4 MLD-TS alternative (`PuschMldEnable = 1`, COMPRESS only)

`phy_ul_pusch_pre_mldts_func` builds a QR decomposition of the regularised channel per RE group, $\big[\mathbf H;\ \sigma\mathbf I\big]=\mathbf Q\mathbf R$. It stores $\mathbf Q^H\mathbf y$ (`pQHyOut`) and the upper-triangular $\mathbf R$ with off-diagonals in 11S6 format (`pROut`). The bbdev accelerator runs a tree-search maximum-likelihood detection. `phy_ul_pusch_post_mldts_func` then re-orders the LLRs per UE ([`bblib_mldts_post_process_5gnr`](../../../sdk/source/phy/lib_mldts_process_5gnr/phy_mldts_post_process_5gnr.cpp)).

### 3.10 PT-RS phase-noise estimation and compensation: `phy_ul_pusch_pn_func` (doc §2.11)

This runs in the NORMAL (FR2) path. In the COMPRESS path, PT-RS parameters (`nPtrsPresent`, `nPtrsTimeDensity`, `nPtrsFreqDensity`, `nPtrsReOffset`, …) are passed into the symbol-processing kernel.

**Input:**

* PT-RS REs extracted by [`ptrs_re_demap_per_ue_per_symbol`](../phy_modules/re_demap/phy_rx_re_demap.h). The PT-RS frequency offset is $k_{RB,ref}=n_{RNTI}\bmod K_{PT\text{-}RS}$ (or modulo $N_{RB}\bmod K_{PT\text{-}RS}$).
* The DMRS channel estimate at the PT-RS subcarriers on the nearest DMRS symbol.
* The PT-RS sequence, which is the DMRS sequence at those REs for CP-OFDM.

**Output:** a common phase error (CPE) per PT-RS symbol, applied to the equalised data.

$$
\Phi_l=\sum_{r}\sum_{j=1}^{N_{PT}}Y(k_j,l,r)\,\big[\hat H(k_j,l',r)\,X_{PT}(k_j)\big]^{*},\qquad
\hat\theta_l=\angle\Phi_l,\qquad
e^{j\hat\theta_l}=\frac{\Phi_l}{|\Phi_l|}
$$

The normalised phasor is what the kernel uses; it avoids the `atan2` call. Compensation is applied to every data symbol, using the nearest PT-RS symbol within the half-slot:

$$
\hat x(k,l)\leftarrow\hat x(k,l)\,e^{-j\hat\theta_{l_{PT}^{*}(l)}}
$$

For DFT-s-OFDM with PT-RS (single layer), the equalised symbol is first transformed back to the time domain (`pTfCOut`, IDFT and scale restore). The PT-RS samples are then taken in groups (`nPtrsGrp`, `nPtrsGrpSamp`), and the same estimator is applied to those time-domain samples.

### 3.11 Transform de-precoding (DFT-s-OFDM)

$$
d(m)=\frac{1}{\sqrt{M}}\sum_{k=0}^{M-1}\hat x(k)\,e^{\,j2\pi km/M},\qquad M=12N_{RB}\ (\le1200\ \text{in code})
$$

This is done with [`bblib_idft_burst_fxp`](../../../sdk/source/phy/lib_dft_idft/phy_dft_idft.cpp). When PT-RS is present it happens before PN compensation (inside `mmse_func`), and otherwise during layer demapping. The fixed-point scale is restored by multiplying by $2^{s_{out}}/\sqrt{M}$ using `mulhrs` followed by a shift.

### 3.12 RE de-mapping and layer de-mapping

* **RE demap** ([`sch_re_demap_per_ue`](../phy_modules/re_demap/phy_rx_re_demap.h)) removes DMRS REs and PT-RS REs (from `nPtrsSymbolIdx`, $K_{PT}$, `nReOffset`, port). On DMRS symbols with data-DMRS multiplexing (`nDataDmrsInter = 1`), it keeps $12 - N_{CDM}\cdot N^{DMRS}_{sc/RB}$ data REs per RB. Output: `pDeRemapDataOut[t]` and `nDeRemapSymbs`.
* **Layer demap** (TS 38.211 §7.3.1.3, inverse): for $\upsilon$ layers, $d(\upsilon i+t)=x^{(t)}(i)$, $t=0\dots\upsilon-1$. For one layer this is a `memcpy`.

### 3.13 Soft-bit (LLR) demapping (doc §2.19, [`bblib_llr_demapping_nn_5gnr`](../../../sdk/source/phy/lib_llr_demapping/phy_llr_demapping_5gnr_nearest_neighbour.cpp) and inside symbol processing)

With unit-energy constellations the biased estimate $\hat x=\beta x+\tilde n$ has

$$
\sigma_{\tilde n}^2=\beta(1-\beta),\qquad \sigma^2_{u}=\tfrac12(\beta-\beta^2)\quad\text{(per real dimension)}
$$

The max-log LLR for bit $i$, with the convention $\lambda=\ln\frac{P(b=0)}{P(b=1)}$ (positive for $b=0$), is

$$
\lambda_i=\frac{1}{2\sigma_u^2}\Big[\min_{a\in\mathcal S_i^{1}}(\Re\hat x-\beta a)^2-\min_{a\in\mathcal S_i^{0}}(\Re\hat x-\beta a)^2\Big]
$$

The same formula is applied on $\Im\hat x$ for the odd bits. The normalisation constants are $c=1/\sqrt2,\ 1/\sqrt{10},\ 1/\sqrt{42},\ 1/\sqrt{170}$ for QPSK, 16QAM, 64QAM and 256QAM.

**π/2-BPSK** ($Q_m=1$; the code uses $(\Re\pm\Im)\gg(\texttt{DE\_MOD\_RIGHT\_SHIFT}+1)$):

$$
\lambda(2i)=\frac{\sqrt2}{1-\beta}\big(\Re\hat x+\Im\hat x\big),\qquad
\lambda(2i+1)=\frac{\sqrt2}{1-\beta}\big(\Im\hat x-\Re\hat x\big)
$$

**QPSK:**

$$
\lambda_0=\frac{2\sqrt2}{1-\beta}\,\Re\hat x,\qquad \lambda_1=\frac{2\sqrt2}{1-\beta}\,\Im\hat x
$$

**16QAM** ($\lambda_0,\lambda_2$ from $\Re\hat x$; $\lambda_1,\lambda_3$ from $\Im\hat x$; $z=\Re\hat x$):

$$
\lambda_0=\frac{1}{2\sigma_u^2}\begin{cases}8\beta c\,(z+\beta c), & z\le-2\beta c\\ 4\beta c\,z,&|z|\le2\beta c\\ 8\beta c\,(z-\beta c),& z>2\beta c\end{cases}
\qquad
\lambda_2=\frac{4\beta c\,(2\beta c-|z|)}{2\sigma_u^2}
$$

**64QAM** ($\lambda_0,\lambda_2,\lambda_4$ from $\Re$; $\lambda_1,\lambda_3,\lambda_5$ from $\Im$):

$$
\lambda_0=\frac{1}{2\sigma_u^2}\begin{cases}
16\beta c(z+3\beta c),& z\le-6\beta c\\
12\beta c(z+2\beta c),& -6\beta c<z\le-4\beta c\\
8\beta c(z+\beta c),& -4\beta c<z\le-2\beta c\\
4\beta c\,z,& |z|\le2\beta c\\
8\beta c(z-\beta c),& 2\beta c<z\le4\beta c\\
12\beta c(z-2\beta c),& 4\beta c<z\le6\beta c\\
16\beta c(z-3\beta c),& z>6\beta c
\end{cases}
$$

$$
\lambda_2=\frac{1}{2\sigma_u^2}\begin{cases}8\beta c(3\beta c-|z|),&|z|\le2\beta c\\4\beta c(4\beta c-|z|),&2\beta c<|z|\le6\beta c\\8\beta c(5\beta c-|z|),&|z|>6\beta c\end{cases}
\qquad
\lambda_4=\frac{1}{2\sigma_u^2}\begin{cases}4\beta c(|z|-2\beta c),&|z|\le4\beta c\\4\beta c(6\beta c-|z|),&|z|>4\beta c\end{cases}
$$

**256QAM** follows the same piecewise-linear max-log structure on $z$ or $|z|$, with breakpoints at multiples of $2\beta c$ up to $14\beta c$. For example, $\lambda_0=4\beta c\,z/(2\sigma_u^2)$ for $|z|\le2\beta c$ and $32\beta c(z+7\beta c)/(2\sigma_u^2)$ for $z\le-14\beta c$. The Intel document lists the 256QAM expressions with an extra common factor of 2.

**Fixed point:** the 16-bit soft value is shifted right by `DE_MOD_RIGHT_SHIFT = 13 - 5 = 8` (NORMAL path), or formatted with `nLlrFxpDecimal` fractional bits (COMPRESS path). It is then saturated to int8 (`nLlrSaturatedBits`). LLRs are written straight into `pDeScrambleOut` in codeword order. The second half-slot is appended at offset `nLen0`.

### 3.14 De-scrambling: `phy_ul_pusch_de_scramble_func` (doc §2.20)

$$
c_{init}=n_{RNTI}\cdot2^{15}+n_{ID},\qquad
\tilde\lambda(i)=\lambda(i)\,\big(1-2c(i)\big),\quad i=0\dots M_{bit}-1
$$

* $c(i)$ is the 38.211 §5.2.1 Gold sequence. $n_{ID}$ is `dataScramblingIdentityPUSCH` if configured, otherwise $N^{cell}_{ID}$.
* $M_{bit}$ = `nDeModBits[0]`.
* The descrambler only flips signs, and works in place. It runs per UE and is split by UE (`LlrRxSplitStruct`).

### 3.15 UCI on PUSCH, LDPC decoding and TB CRC

1. **Data/control demultiplexing** (38.212 §6.2.7, inverse) separates HARQ-ACK, CSI part 1, CSI part 2 and UL-SCH LLRs (`phy_ul_pusch_uci_decode_func`). Each UCI stream goes through rate de-matching and channel decoding (polar, RM or repetition) and its CRC check. The CSI-2 length is derived from the decoded CSI-1.
2. **UL-SCH** goes to the bbdev accelerator through `gnb_io_fec_ul_bbdev_ldpc_decoder_run_func`. The accelerator does code-block de-concatenation, bit de-interleaving, bit de-selection from $k_0(rv)$, HARQ soft combining, LDPC decoding and per-CB CRC.
3. `phy_ul_bbdev_pusch_tb_func` checks the TB CRC (`gnb_io_fec_ul_bbdev_ldpc_decoder_crc`), sets `nTbCrcOut` (forced to 0 if there was an O-RAN packet error), updates throughput statistics, and calls `nr5g_gnb_phy2mac_api_pusch_rxpdu_proc`.

---

## 4. Uplink Measurements and Reporting

All of the following are converted in `api/phy2mac/gnb_phy2mac_api_rxsdu_proc.c`. Reported values use 1/256 dB steps (`SINR_STEP_SIZE = 256`) unless noted.

### 4.1 Digital-to-RF power calibration

Most power reports go through

$$
P_{\text{dBm}}=10\log_{10}P_{lin}+\underbrace{10\log_{10}2^{26}-20\log_{10}32767+\frac{\texttt{Rx0DbfsRadioPower}}{65536}}_{\text{cal}}
$$

(`sigPwrRfCalibration(10log10(P)+30)`, whose internal −30 cancels the +30). The $2^{26}$ term undoes the Q26 power scaling. $20\log_{10}32767$ references power to int16 full scale, so the first part of the expression is power in dBFS. `Rx0DbfsRadioPower` is the RF power in dBm at 0 dBFS (Q16). In test builds without `PHY_APP`, cal is fixed at −99.7465 dB. Values below `PWR_INVALID` are clamped.

### 4.2 SNR (pre-equalisation, DMRS based): `CRC.indication.nSNR`

For UE $u$ with layers $\mathcal T_u$:

$$
\mathrm{SNR}_u[\text{dB}]=10\log_{10}\Big(\frac{1}{|\mathcal T_u|}\sum_{t\in\mathcal T_u}\overline{\mathrm{SNR}}_t\Big)-20\log_{10}\beta_{DMRS},
\qquad \texttt{nSNR}=\operatorname{round}(256\cdot\mathrm{SNR}_u)
$$

The $\beta_{DMRS}$ term removes the DMRS boost, so the value refers to the data REs. If the result is NaN or Inf, `PUSCH_MIN_SNR` is reported.

* **Smoothing (per UE):** $\overline{\mathrm{SNR}}_n=0.125\,\mathrm{SNR}_n+0.875\,\overline{\mathrm{SNR}}_{n-1}$. The first value is $\mathrm{SNR}$ if the CRC passed, and $\mathrm{SNR}/2$ if it failed.
* **DTX:** if $\overline{\mathrm{SNR}}\le -6\,\text{dB}-20\log_{10}\beta_{DMRS}$ (or the SNR is invalid while the CRC passes) for more than 10 consecutive slots, the UE is declared lost.
* **TO-estimate confidence** (`nConfLvl`): 1 if $10\log_{10}(N_R N_{RB}N_L N_{DMRS})+\mathrm{SNR}_{dB}>20$, else 0.

### 4.3 Post-equalisation SINR: `CRC.indication.nPostSNR[t]`

$$
\texttt{nPostSNR}[t]=\operatorname{round}\big(256\cdot10\log_{10}\texttt{fPostSINR}[t]\big),\qquad
\texttt{fPostSINR}[t]=\overline{\beta_t/(1-\beta_t)}\ \ (\S3.9.3)
$$

In both paths only the first channel (or data) symbol is used, not the whole slot. In the DFT-s-OFDM path the value is $\bar\beta/(1-\bar\beta)$, with β averaged over subcarriers first (§3.9.3).

This is the SINR that actually reaches the demapper. It includes the MIMO/IRC combining gain and inter-layer and residual interference, so it is the better input for link adaptation. The pre-equalisation SNR in §4.2 is per receive antenna and per layer.

### 4.4 Received DMRS power (reported as signal power): `nSigPwr[r]`, `nTotalSigPwr`

This value comes from the CE power bits. Despite its name, it is **not** a desired-signal (RSRP) measurement. As §3.4.4 shows, the power bits are the **received** DMRS power per RE (signal + interference + noise), **divided** by $\beta_{DMRS}^2$, with AGC restored. They are also identical for all layers of a CDM group:

$$
P_{sig}(r)=\frac{1}{N_L^{u}}\sum_{t\in\mathcal T_u}\frac{2^{2b(t,r)}}{2^{26}}\;=\;\frac{P_{rx}(r)}{2^{26}}\ \text{(when all the UE's layers share one CDM group)}
$$

$$
\texttt{nSigPwr}[r]=256\cdot P_{\text{dBm}}\big(P_{sig}(r)\big),\qquad
\texttt{nTotalSigPwr}=256\cdot P_{\text{dBm}}\Big(\sum_r P_{sig}(r)\Big)
$$

* `nSigPwr[r]` is the per-antenna, in-allocation received DMRS power per RE, referred to the data-RE power level. It behaves like an in-allocation RSSI more than an RSRP.
* `nTotalSigPwr` is the sum over antennas.
* The noise-corrected desired-signal power $P_S(t,r)$ from §3.4.4 is used only for the SNR. It is not reported here.

### 4.5 RSSI

**(a) Wideband RSSI task** (`phy_ul_rssi_measurement_task`, scheduled by `RSSI` measurement config):

$$
P_{RSSI}(l,s,r)=\frac{1}{N_{RB,s}}\sum_{k\in s}|Y(k,l,r)|^2\quad(\text{linear power per RB per antenna, includes signal + interference + noise})
$$

For non-massive-MIMO configurations the value is summed over antennas, $P_{RSSI}(l,s)=\sum_r P_{RSSI}(l,s,r)$. For massive MIMO it is reported per RU stream. The report is $\texttt{nRssi}=256\cdot P_{\text{dBm}}(P_{RSSI})$ in `RSSI.indication`, per symbol, subband and stream. If the O-RAN packets were bad, the value $-1$ (invalid) is reported.

**(b) DMRS-symbol received power, dBFS** (`nDMRSPwrReportEna`):

$$
P_{DMRS}=\frac{1}{N_R N_{DMRS} 12N_{RB}}\sum_{r,l,k}|Y_{RS}(k,l,r)|^2,\qquad
\texttt{nDMRSPwrDBFS}=100\cdot\big(10\log_{10}P_{DMRS}-10\log_{10}2^{30}\big)
$$

This is reported in units of 0.01 dB. It is an in-allocation RSSI measured on the DMRS symbols.

### 4.6 Timing advance: `nTA`, `nTANanos`

From the per-layer TO $\bar\tau_t$ (in FFT samples), the layer with the highest SNR, $t^{*}$, is used:

$$
\texttt{nTA}=\Big\lfloor\bar\tau_{t^{*}}\cdot\frac{128}{N}\Big\rfloor+31,\qquad
\texttt{nTANanos}=\bar\tau_{t^{*}}\cdot\frac{10^{9}}{15\cdot10^{3}\cdot N\cdot2^{\mu}}
$$

The factor $128/N$ converts samples into TA-command units, because one TA step is $16\cdot64\,T_c/2^\mu$ and one FFT sample is $1/(N\Delta f)$; the ratio of the two is $128/N$. The offset 31 (`UL_TA_ZERO`) centres the 6-bit TA command (0…63), so 31 means no adjustment. `nPuschTimeOffset` (layer 0, raw samples) is also stored per connected UE.

### 4.7 Interference-plus-noise power per RBG: `nIntfNoisePwr[rbg]`

$$
\texttt{nIntfNoisePwr}[g]=256\cdot P_{\text{dBm}}\big(\texttt{fNoiseIntfPwr}[g]\big),\qquad
\texttt{fNoiseIntfPwr}[g]=s_{AGC}\cdot\operatorname*{mean}_{k\in g,\,r}|e(k,l,r)|^2
$$

This is the per-4-RB "IoT" map that the MAC uses for interference-aware scheduling.

### 4.8 Frequency offset and Doppler

* `fEstCfo[t]` (Hz) and `fEstAngleOffset[t]` (rad per symbol) are kept per UE group and used internally by the equaliser.
* `fEstDopplerShift[t]` (Hz, Kalman-filtered) is kept per UE group. It selects the time-interpolation granularity and is available to the stats/BEVO interfaces.

---

## 5. Fixed-Point and Configuration Constants

| Constant | Value | Meaning |
|---|---|---|
| `PHY_RX_DATA_FIXED_POINT` | 13 | IQ samples are Q13 (int16) |
| `MMSE_NOISE_POWER` | 26 | $\sigma^2$ and power scaled by $2^{26}$ (Q26) |
| `DE_MOD_RIGHT_SHIFT` | 8 | LLR scaling shift (NORMAL path) |
| DMRS scale | 14 | DMRS base sequence Q14 |
| `agcTarget` | config (doc default $2^{13}$) | AGC target amplitude |
| `IrcEnableThreshold` | config (dB) | SNR below this enables IRC |
| `nEnableIrc` | 0 = auto, 1 = per config, 2 = force (2-SC), 3 = force | IRC mode |
| `IrcDynamicFlag` | 0/1 | Per-4-RB MMSE/IRC switch |
| `PuschNoiseScale` | config (2 on Windows build) | $\sigma^2$ multiplier, multi-layer SU at $\mu=0$ |
| `interpTable` | 0/1/2 | Time-interpolation selection table |
| `RNN_REFINE` | 1 | $\mathbf R_{nn}$ refinement with fixed α = 0.25; mode 2 uses an INR-adaptive α (§3.6) |
| `UL_TA_ZERO` | 31 | TA command zero point |
| `SNR_AVG_COEFFICIENT` | 0.125 | SNR EMA weight |
| `PUSCH_SNR_DTX_THRESHOLD` / `_CNT_` | −6 dB / 10 | UE DTX / loss detection |
| `SINR_STEP_SIZE` | 256 | 1/256 dB report resolution |

---

## 6. End-to-End Data Flow (one UE group, one half-slot)

```
pCEin[port][dmrsSym]  (Q13, after BFP decompression)
   │  AGC: g = round(log2(rms/target))
   ▼
LS:  H~ = Y * conj(X_RS)/beta  →  OCC de-spread (freq [+time])  →  TO est τ, comp e^{+j2πkτ/N}
   ▼
MMSE-CE (2-step Wiener, 4-RB groups, sinc corr, SNR0=30dB → SNR_est) → TO de-comp
   │──► residual e = Y - Σ X H  ──► σ² (fSigma2Mean), IpN/RBG (fNoiseIntfPwr), pCEIpNOut ──► R_nn^-1 (IRC)
   │──► SNR(t,r), power bits b(t,r)                                 ──► nSNR, nSigPwr
   │──► Σ H(l1) H*(l0) → φ, f_CFO ; R0,R1 → f_D (Kalman)            ──► FO comp, interp choice
   ▼
pCeOut[t][r]  (all SCs, CE symbols)
   ▼
for each data symbol l:  H(l) = TimeInterp(e^{-jφ l_j} H(l_j)) · e^{+jφ l}
   ▼
W = (HᴴH+σ²I)^-1 Hᴴ   or   (Hᴴ R⁻¹ H + I)^-1 Hᴴ R⁻¹        x̂ = W y ;  β = diag(WH) ;  SINR = β/(1-β)
   ▼
[FR2] PT-RS CPE: θ_l = ∠Σ Y·(H X_PT)* ; x̂·e^{-jθ}       [TP] IDFT
   ▼
RE demap → layer demap → LLR (max-log, scaled by 1/(1-β), σ_u²=½β(1-β)) → int8
   ▼
descramble λ·(1-2c), c_init = RNTI·2^15 + n_ID  →  UCI demux/decode  |  LDPC (bbdev) → TB CRC → MAC
```
