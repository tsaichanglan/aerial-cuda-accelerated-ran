# PUSCH Pipeline Front End: Figure-to-Code Map

This maps each block of the figure "Graph Diagram of the PUSCH Pipeline Front End" to the cuPHY code that implements it. The figure comes from the cuPHY Developer Guide, cuPHY Components page; a redrawn copy is in [PUSCH_pipeline_front_end_graph.drawio](PUSCH_pipeline_front_end_graph.drawio).

All paths are relative to the repository root. Line numbers refer to the current `main` branch.

## Where the pipeline is driven

| Role | Code |
|---|---|
| Pipeline object (setup, run) | `PuschRx` in [cuPHY/src/cuphy_channels/pusch_rx.cpp](../cuPHY/src/cuphy_channels/pusch_rx.cpp) |
| Stream-mode launch order | `PuschRx::fullSlotKernelLaunch()` (line 9384) |
| Graph-mode wiring of the front end | `PuschRx::buildFrontEndStage()` (line 4113) and `PuschRx::buildSoftDemapStage()` (line 3900), called from `PuschRx::createFullSlotGraph()` (line 5074) |
| Early-HARQ (sub-slot) variant | `PuschRx::subSlotKernelLaunch()` (line 9183), `createEarlyHarqGraph()` (line 2951) |

## Block-to-code table

| Figure block | Directory / host class | GPU kernel(s) (file:line) | Reads | Writes |
|---|---|---|---|---|
| **PUSCH IQ Data buffer** | Filled by the cuPHY-CP order/decompression kernels (`cuPHY-CP/cuphydriver/src/uplink/`) | — | fronthaul U-plane | `DataRx` `[NF_cell, 14, nRxAnt]` per cell |
| **LS Channel Estimation** | `cuPHY/src/cuphy/ch_est/`, class `puschRxChEst` (kernel selection in `puschRxChEstKernelBuilder`) | Multi-stage algorithm, stage ①: `windowedChEstPreNoDftSOfdmKernel` ([ch_est.cu:1811](../cuPHY/src/cuphy/ch_est/ch_est.cu#L1811)). Legacy algorithm: LS is fused into the MMSE kernel (next row). | `DataRx` (DMRS symbols) | `DmrsLSEst`, `DmrsAccum` |
| **MMSE Channel Estimation** | same | Multi-stage stage ②: `chEstFilterNoDftSOfdmDispatchKernel` ([ch_est.cu:1281](../cuPHY/src/cuphy/ch_est/ch_est.cu#L1281)), or `chEstFilterPrgNoDftSOfdmDispatchKernel` ([:1592](../cuPHY/src/cuphy/ch_est/ch_est.cu#L1592)) with per-PRG estimation.<br>Legacy (LS + MMSE in one kernel): `windowedChEstNoDftSOfdmKernel` ([:425](../cuPHY/src/cuphy/ch_est/ch_est.cu#L425)) for ≥ 4 PRBs, `smallChEstNoDftSOfdmKernel` ([:3638](../cuPHY/src/cuphy/ch_est/ch_est.cu#L3638)) for 1–3 PRBs. DFT-s-OFDM versions: `windowedChEstKernel`, `smallChEstKernel`.<br>Alternatives: RKHS `puschRkhsChEstKernel` ([:4387](../cuPHY/src/cuphy/ch_est/ch_est.cu#L4387)); TensorRT `prepareChestMlInputsKernel` → engine → `extractChestMlOutputsKernel` ([ch_est_trtengine_pre_post_converion.cu](../cuPHY/src/cuphy/ch_est/ch_est_trtengine_pre_post_converion.cu)) | `DmrsLSEst`, `DmrsAccum` (multi-stage) or `DataRx` (legacy) | `HEst` `[nRxAnt, nLayers, NF, NH]` |
| **Noise and Interference Covariance Estimation** | `cuPHY/src/cuphy/pusch_noise_intf_est/`, class `puschRxNoiseIntfEst` | `noiseIntfEstKernel` ([pusch_noise_intf_est.cu:1164](../cuPHY/src/cuphy/pusch_noise_intf_est/pusch_noise_intf_est.cu#L1164)). Its body is `noiseIntfEstNoDftSOfdmKernelInner` (line 493): it forms the residual on the DMRS REs (received minus H × pilots) and accumulates the covariance Rww per PRB. | `DataRx`, `HEst` | Rww (internal), `NoiseVarPreEq` |
| **Shrinkage and Whitening** | Shrinkage and the whitening matrix: same kernel as above. Applying the whitening: `cuPHY/src/cuphy/channel_eq/` | Shrinkage is in `noiseIntfEstKernel` (lines ~1005–1040), only for `PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW` / `_OAS`: nCov = (1−ρ)·Rww + ρ·tr(Rww)/nAnt·I.<br>The whitening matrix Lw⁻¹ is computed by `cholFactor` (line 317), `triLInv` (line 386) and `cholFactorInv` (line 441), giving the inverse Cholesky factor of Rww.<br>The whitening is applied inside the MMSE-IRC coefficient kernel (`eqMmseIrcCoefCompLowMimoKernel`, `channel_eq.cu` ~line 2840, which reads `LwInv`). | Rww | `LwInv` `[nRxAnt, nRxAnt, nPrb]` |
| **CFO estimation and CFO estimate Averaging** | `cuPHY/src/cuphy/cfo_ta_est/`, class `puschRxCfoTaEst` | `cfoTaEstLowMimoKernel` ([cfo_ta_est.cu:675](../cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu#L675)). It takes the phase of the correlation between `HEst` at different DMRS times and averages it over PRBs and antennas. The SINR-weighted average is used when `enableWeightedAverageCfo` is set (line 586). | `HEst`, `SinrPreEq` (weighted mode) | `CfoEst` `[14, nUes]`, `CfoPhaseRot`, `CfoHz` |
| **Timing Offset Estimation and Averaging** | same | same kernel, `cfoTaEstLowMimoKernel` (phase slope of `HEst` across subcarriers) | `HEst` | `TaEst`, `TaPhaseRot` |
| **Channel Equalization Coefficients** | `cuPHY/src/cuphy/channel_eq/`, class `puschRxChEq` (`eqMmseCoefCompLowMimo/HighMimo/MassiveMimo`, lines 5263–5398) | `eqMmseCoefCompLowMimoKernel` ([channel_eq.cu:3330](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L3330)). It dispatches to an RZF/MMSE variant (reads `CfoEst`, ~line 2240) or the MMSE-IRC variant (~line 2840).<br>`eqMmseCoefCompHighMimoKernel` ([:1734](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L1734)).<br>`eqMmseCoefCompMassiveMimoKernel_v1` ([:1786](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L1786)). | `HEst`, `LwInv`, `NoiseVarPreEq`, `CfoEst` | `EqCoef` `[nRxAnt, 12, nLayers, nPrb, NH]`, `ReeDiagInv` `[12, nLayers, nPrb, nTimeChEq]` |
| **Equalize Data (ChEqApply + SoftDemapping)** | same | `eqMmseSoftDemapKernel` ([channel_eq.cu:5044](../cuPHY/src/cuphy/channel_eq/channel_eq.cu#L5044)): equalization and LLR demapping in one kernel; body `eqMmseSoftDemapKernel_v4` (line 3583).<br>Variants:<br>• `eqMmseSoftDemapKernel_64R` (line 5217), 64 receive antennas;<br>• `eqMmseEqualizationKernel` (line 5144), equalize-only mode;<br>• `eqMmseSoftDemap72eKernel` (line 5168), split 7.2e;<br>• DFT-s-OFDM: `bluestein_workspace_kernel` / `bluestein_Idft_kernel`, then `eqMmseSoftDemapAfterDftKernel` (line 5192). | `DataRx`, `EqCoef`, `ReeDiagInv`, `CfoEst` | `LLR` `[8, nLayers, NF, Nd]` (+ `LLRCdm1`, `DataEq` debug) |
| **Estimate Noise Variance** → preEqualization Noise Power | `pusch_noise_intf_est/` | `noiseIntfEstKernel` (above) | `DataRx`, `HEst` | `NoiseVarPreEq` |
| **Estimate Noise Variance** → postEqualization Noise Power | `cuPHY/src/cuphy/pusch_rssi/`, class `puschRxRssi` | `rsrpMeasKernel_v1` ([pusch_rssi.cu:733](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L733)) or `rsrpMeasKernel_v2` (line 1419), derived from `ReeDiagInv` | `ReeDiagInv`, `NoiseVarPreEq` | `NoiseVarPostEq` |
| **RSRP Estimation and Averaging** | `pusch_rssi/` | `rsrpMeasKernel_v1` / `_v2` (above). `rsrpMeasWithoutPostEqMeasKernel_v1` (line 1046) skips the post-equalization measures. Early-HARQ: `sinrMeasEarlyHarqKernel_v1` (line 1235). | `HEst` | `Rsrp` `[nUes]` |
| **SNR Estimation** | `pusch_rssi/` | same RSRP kernels | `Rsrp`, `NoiseVarPreEq`, `ReeDiagInv` | `SinrPreEq`, `SinrPostEq` `[nUes]` |
| **RSSI Estimation and Averaging** | `pusch_rssi/` | `rssiMeasKernel` ([pusch_rssi.cu:229](../cuPHY/src/cuphy/pusch_rssi/pusch_rssi.cu#L229)) | `DataRx` | `Rssi` `[nUeGrps]`, `RssiFull` `[14, 64, nUeGrps]` |
| **Proceed to decoding of CSI-P1 and SCH payload** | `uci_on_pusch/`, `rate_matching/`, `ldpc/`, `crc/` | `uciOnPuschSegLLRs0`, de-rate-matching, LDPC decoder, CRC (launched later in `fullSlotKernelLaunch`) | `LLR` | TBs, UCI |

## How the figure's arrows map to launch order

The figure shows data flow. The code orders the kernels as follows; see `buildFrontEndStage` for graph mode and `fullSlotKernelLaunch` for stream mode.

1. **Channel estimation:** one launch per DMRS time position and per kernel configuration. With the multi-stage algorithm, stage ① launches first, then stage ②.
2. **Noise/interference estimation and CFO/TA estimation:** they run in parallel. Both depend only on channel estimation.
   - Noise/interference runs only if SINR measurement is enabled or the equalizer is an MMSE type.
   - With `enableWeightedAverageCfo`, the order becomes a chain instead: noise → RSRP → CFO/TA, because CFO needs the SINR weights.
3. **Equalizer coefficients:** depend on channel estimation, noise and CFO. One launch per equalizer time position (up to 4 with TDI) and per configuration.
4. **Equalization and soft demapping:** one kernel that depends on all coefficient kernels. With DFT-s-OFDM it is followed by the IDFT and after-DFT demapping kernels.
5. **RSSI and RSRP/SNR:** these are measurements and do not feed the data path, except for the weighted-CFO case. In stream mode they are launched after the decoders (`fullSlotKernelLaunch`, lines 9674–9714).

## Differences between the figure and the code

- **Fused kernels.** The figure draws LS and MMSE estimation as separate blocks, and coefficient computation separate from equalization. In the code:
  - the legacy channel-estimation kernel does LS and MMSE in one kernel;
  - equalization and soft demapping are a single kernel (`eqMmseSoftDemapKernel`).
- **Shrinkage and whitening are split** across two kernels. The covariance is shrunk and Lw⁻¹ is computed in `noiseIntfEstKernel`. The whitening is applied later, inside the MMSE-IRC coefficient kernel.
- **One kernel produces several outputs:**
  - CFO and timing offset come from `cfoTaEstLowMimoKernel`;
  - RSRP, post-equalization noise variance and SNR come from `rsrpMeasKernel_v1`/`_v2`.
- **Pre-equalization noise power has a different source.** "Estimate Noise Variance" is one block in the figure, but pre-equalization noise power is written by the noise/interference kernel (`NoiseVarPreEq`). Only the post-equalization value comes from the RSRP kernel.
- **"DMRS[0] received"** is the early-HARQ trigger. The sub-slot path (`subSlotKernelLaunch`) runs channel estimation and the stages after it on the first DMRS symbol(s) as soon as they arrive. The full-slot pass then continues from the next DMRS position.
- **Not shown in the figure:** descrambling, which is part of channel estimation, and the TensorRT/RKHS estimator alternatives.
