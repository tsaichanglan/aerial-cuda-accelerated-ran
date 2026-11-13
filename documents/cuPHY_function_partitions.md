# cuPHY Function Partitions

Scope: the `cuPHY/` directory only (the GPU L1 library, its channel pipelines, tests, examples and utilities). Control plane (`cuPHY-CP`), cuMAC and pyAerial are out of scope.

Every directory under `cuPHY/src/` is assigned to exactly one partition. Paths are relative to `cuPHY/`; `cuphy/` means `cuPHY/src/cuphy/`. 3GPP references are to the specification as a whole (TS 38.211 physical channels and modulation, TS 38.212 multiplexing and channel coding, TS 38.213/38.214 physical-layer procedures).

## How the partitions fit together

- **P1 (channel pipelines)** is the top layer: one object per channel, which the PHY driver calls once per slot.
- **P2–P10** are the signal-processing functions those pipelines are built from.
- **P11–P14** are the shared platform, I/O, validation and build support that everything else uses.

```
P1  Channel pipelines (PDSCH, PDCCH, SSB, CSI-RS, BFW, PUSCH, PUCCH, PRACH, SRS)
     │ built from
     ├─ Downlink:  P2 DL signal generation
     ├─ Uplink:    P3 UL receiver front end · P6 UCI · P7 PUCCH · P8 PRACH · P9 SRS + beamforming
     ├─ Both:      P4 Channel coding (LDPC, Polar, small-block, CRC, rate matching)
     │             P5 Sequences and scrambling
     ├─ AI:        P10 AI/ML inference
     └─ Support:   P11 Core platform · P12 Profiling + logging · P13 Data I/O · P14 Validation + build
```

## Partition list

### P1 — Channel pipelines

Per-channel objects that validate parameters, allocate buffers, batch all cells of a cell group, and launch the kernels of the other partitions (stream or CUDA-graph mode).

| Sub-function | Code |
|---|---|
| PDSCH transmit | `src/cuphy_channels/pdsch_tx.*` |
| PDCCH transmit | `src/cuphy_channels/pdcch_tx.*`, `pdcch_polar_cidx2uidx_lut_cpu.h` |
| SSB transmit | `src/cuphy_channels/ssb_tx.*` |
| CSI-RS transmit | `src/cuphy_channels/csirs_tx.*` |
| Beamforming weights | `src/cuphy_channels/bfw_tx.*` |
| PUSCH receive (incl. early-HARQ sub-slot path, work cancellation) | `src/cuphy_channels/pusch_rx.*`, `pusch_utils.hpp`, `cuphy/pusch_start_kernels.*`, `cuphy/pusch_start_kernels_interface.hpp` |
| PUCCH receive | `src/cuphy_channels/pucch_rx.*` |
| PRACH receive | `src/cuphy_channels/prach_rx.*` |
| SRS receive | `src/cuphy_channels/srs_rx.*` |
| UE-side / test signals (CSI-RS receive, SRS transmit) | `src/cuphy_channels/csirs_rx.*`, `srs_tx.*`, `cuphy/csirs_rx/`, `cuphy/srstx/` |
| Public channel API | `cuphy/cuphy_api.h`, `src/cuphy_channels/cuphy_channels.hpp` |

### P2 — Downlink signal generation

Produces frequency-domain resource elements for each downlink channel. TS 38.211.

| Sub-function | Code |
|---|---|
| Modulation mapping (QPSK–256QAM) | `cuphy/modulation_mapper/` |
| PDSCH DMRS, layer mapping, precoding, RE mapping, CSI-RS RE skipping | `cuphy/pdsch_dmrs/` |
| PDCCH: rate matching, scrambling, CCE-to-REG mapping, DMRS | `cuphy/pdcch/` |
| SSB: PSS/SSS, PBCH payload, PBCH DMRS, SSB mapping | `cuphy/ss/` |
| CSI-RS sequence generation, CDM, mapping | `cuphy/csirs/` |

### P3 — Uplink receiver front end (PUSCH)

Turns received frequency-domain IQ into LLRs, plus measurements. TS 38.211, TS 38.214.

| Sub-function | Code |
|---|---|
| DMRS channel estimation (legacy MMSE, multi-stage MMSE + delay est., RKHS, LS) | `cuphy/ch_est/` (except TensorRT path, see P10), `cuphy/channel_est/` |
| Noise and interference estimation | `cuphy/pusch_noise_intf_est/` |
| CFO and timing-offset estimation and correction | `cuphy/cfo_ta_est/` |
| Equalizer coefficients and equalization (RZF, MMSE, MMSE-IRC, shrinkage), DFT-s-OFDM IDFT | `cuphy/channel_eq/` |
| Soft demapping to LLRs | `cuphy/soft_demapper/` |
| RSSI / RSRP / SINR measurements | `cuphy/pusch_rssi/` |
| Descrambling | `cuphy/descrambling/` |

### P4 — Channel coding

Encoders, decoders and rate matching shared by downlink and uplink. TS 38.212.

| Sub-function | Code |
|---|---|
| LDPC encoder and decoder (BG1/BG2, TB and CB modes) | `cuphy/error_correction/`, `cuphy/ldpc/`, `cuphy/cuphy_ldpc.cpp` |
| Downlink rate matching (incl. LBRM) | `cuphy/dl_rate_matching/` |
| Uplink de-rate-matching and HARQ combining | `cuphy/rate_matching/` |
| CRC attach and check | `cuphy/crc/` |
| Polar encoder (PDCCH, PBCH) | `cuphy/polar_encoder/` |
| Polar decoder (CRC-aided list decoding), codeword tree types, segmentation / de-rate-match / de-interleave | `cuphy/polar_decoder/`, `cuphy/comp_cwTreeTypes/`, `cuphy/polar_seg_deRm_deItl/` |
| Small-block decoders: Simplex (≤ 2 bits) | `cuphy/simplex_decoder/` |

Reed-Muller decoding (3–11 bits) lives in `cuphy/pucch_receiver/rm_decoder.*` and is listed under P7, where its code sits.

### P5 — Sequences and scrambling

| Sub-function | Code |
|---|---|
| Gold-sequence generation and lookup tables | `cuphy/Gold_sequence/` |
| Low-PAPR base sequences (DMRS for DFT-s-OFDM, PUCCH, SRS) | `cuphy/constants.hpp` |

`Gold_sequence/` is about 222k lines, but it is two lookup-table headers, not logic.

### P6 — UCI processing

Uplink control information carried on PUSCH and PUCCH formats 2/3/4. TS 38.212.

| Sub-function | Code |
|---|---|
| UCI on PUSCH: LLR segmentation (HARQ / CSI-1 / CSI-2), de-rate-matching, CSI-2 size control, DTX detection | `cuphy/uci_on_pusch/` |
| PUCCH F2/F3/F4 UCI segmentation | `cuphy/pucch_F234_uci_seg/` |

### P7 — PUCCH receivers

TS 38.211, TS 38.213.

| Sub-function | Code |
|---|---|
| Format 0 (sequence correlation) | `cuphy/pucch_F0_receiver/` |
| Format 1 (channel estimation, OCC despreading) | `cuphy/pucch_F1_receiver/` |
| Format 2 front end | `cuphy/pucch_F2_front_end/` |
| Format 3 front end (incl. CSI-P2 control) | `cuphy/pucch_F3_front_end/` |
| Common PUCCH receiver, LUT generation, Reed-Muller decoder | `cuphy/pucch_receiver/` |

### P8 — PRACH receiver

Preamble detection for formats 0 and B4 (Zadoff-Chu correlation via FFT, non-coherent combining, peak detection). TS 38.211.

| Sub-function | Code |
|---|---|
| PRACH receiver kernels | `cuphy/prach_receiver/` |

### P9 — SRS and beamforming

| Sub-function | Code |
|---|---|
| SRS channel estimation (MMSE, RKHS) | `cuphy/srs_chEst/` |
| Beamforming weight computation (regularized zero-forcing) | `cuphy/bfc/` |
| Beamforming-weight block-floating-point compression and packing | `src/compression_decompression/` |

### P10 — AI/ML inference in the PHY

| Sub-function | Code |
|---|---|
| TensorRT engine wrapper | `cuphy/trt_engine/` |
| TensorRT-based channel estimator and the channel-estimator factory that selects it | `cuphy/ch_est/trtengine_chest.cpp`, `cuphy/ch_est/` factory and YAML loader (`chest_factory.hpp`, `ch_est_yaml_loader.cpp`) |
| Model conversion example (PyTorch → ONNX → TensorRT) | `examples/ch_est/torch_to_trt_chest_example.py`, `examples/ch_est/chest_trt.yaml` |

P10 shares the `ch_est/` directory with P3. The boundary is the factory interface: P3 provides the classic estimators and P10 provides the TensorRT one.

### P11 — Core platform

Shared types and runtime used by every other partition.

| Sub-function | Code |
|---|---|
| Public low-level API and its implementation | `cuphy/cuphy.h`, `cuphy/cuphy.cpp`, `cuphy/cuphy.map` (exported symbols) |
| Tensor descriptors and tensor operations (convert, elementwise, fill, reduction, tile) | `cuphy/tensor_*`, `cuphy/convert_tensor.*`, `cuphy/type_convert.hpp`, `cuphy/variant.*` |
| Context, device, error handling | `cuphy/cuphy_context.*`, `cuphy/device.*`, `cuphy/cuphy_err.hpp`, `cuphy/cuphy_internal.h`, `cuphy/cuphy_common.cpp` |
| Kernel and math utilities | `cuphy/cuphy_kernel_util.cuh`, `cuphy/math_utils.cuh`, `cuphy/cuphy_complex_ops.cuh`, `cuphy/cuphy_clmad_util.cuh`, `cuphy/utils.cuh`, `cuphy/common_utils.hpp`, `cuphy/cuphy_utils.hpp`, `cuphy/copy_kernel.cu`, `cuphy/empty_kernels.*` |
| Random-number generation (noise for simulation) | `cuphy/rng/` |
| Shared utilities library | `util/aerial_utils/` |

### P12 — Profiling and logging

| Sub-function | Code |
|---|---|
| Kernel timing / performance instrumentation | `cuphy/cuphy_pti.*`, `cuphy/cuphy_pti_internal.hpp` |
| CUPTI integration | `cuphy/cupti_helper.*` |
| Logging library | `nvlog/` |

### P13 — Data I/O

| Sub-function | Code |
|---|---|
| HDF5 test-vector and debug-dump reading/writing | `src/cuphy_hdf5/` |

### P14 — Validation, tools and build

| Sub-function | Code |
|---|---|
| Unit and component tests | `test/` (one subdirectory per function; runners `cuphy_unit_test.sh`, `test_summary.py`) |
| Examples and standalone benches, including BLER/performance curve generators | `examples/` |
| Analysis and support tools (LDPC, soft-demapper, MATLAB helpers, system checks, perf collection) | `util/ldpc/`, `util/soft_demapper/`, `util/matlab/`, `util/cuBB_system_checks/`, `util/pusch_perf_collect.py`, `util/ptxas_lcheck.py` |
| Build system | `CMakeLists.txt` files, `cmake/` |
| Documentation | `docs/`, `README.md`, `AGENTS.md` |
