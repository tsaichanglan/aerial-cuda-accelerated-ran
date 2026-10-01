# Cell Group in cuPHY PDSCH and PUSCH

> Source paths below are relative to the root of the [aerial-cuda-accelerated-ran](https://github.com/NVIDIA/aerial-cuda-accelerated-ran) repository.

In cuPHY, a **cell group** is the set of cells whose PDSCH (or PUSCH) work for one slot is handled together by a single cuPHY channel object, in one setup call and one run call. It is a batching unit for the GPU, not a radio concept.

It is **not** the 3GPP "cell group" (the Master and Secondary Cell Groups used in dual connectivity and carrier aggregation). The name is the same but the meaning is unrelated.

## Why it exists

Launching separate GPU kernels for every cell each slot would waste time. cuPHY instead combines all the cells' work, such as every transport block across all cells, into shared arrays and runs one batched pipeline.

The PHY driver collects the slot's per-cell commands into one set of cell-group parameters and hands them to `PdschTx` or `PuschRx`. For downlink this happens in `cuPHY-CP/cuphydriver/src/downlink/phypdsch_aggr.cpp`.

## Structure

Both channels use the same layout: a set of flat arrays that link to each other by index.

```
Cell group (one per channel per slot)
 ├─ cells[nCells]        slot number, symbol allocation, index to the cell's static config
 ├─ UE groups[nUeGrps]   UEs sharing the same PRBs/symbols (MU-MIMO); each points to its cell
 ├─ UEs[nUes]            DMRS ports, layers, RNTI, scrambling IDs; each points to its UE group
 └─ (PDSCH only) codewords[nCws], CSI-RS params, precoding matrices
```

- **PDSCH:** `cuphyPdschCellGrpDynPrm_t`, defined in `cuPHY/src/cuphy/cuphy_api.h`. Besides cells, UE groups and UEs, it has:
  - codeword parameters (`nCws`, `pCwPrms`)
  - CSI-RS parameters, used for rate matching around CSI-RS resource elements (`nCsiRsPrms`, `pCsiRsPrms`)
  - precoding matrices (`nPrecodingMatrices`, `pPmwPrms`)
- **PUSCH:** `cuphyPuschCellGrpDynPrm_t`, defined in `cuPHY/src/cuphy/cuphy_api.h`. It has only cells, UE groups and UEs. Per-UE settings such as UCI and HARQ are attached to each UE entry. Each cell's received IQ samples arrive as a separate tensor in `cuphyPuschDataIn_t.pTDataRx`, selected by `cellPrmDynIdx`.

## Static vs. per-slot parameters

| When | Parameter | Meaning |
|---|---|---|
| Object creation (static) | `nCells` (PDSCH) / `nMaxCells` (PUSCH) | All cells the object may ever serve, each with its antennas, bandwidth and numerology (`cuphyCellStatPrm_t`) |
| Object creation (static) | `nMaxCellsPerSlot` | Largest number of cells in a single slot; sets how much GPU memory is reserved |
| Each slot (dynamic) | `nCells` in the cell-group struct | Cells actually active this slot; must be ≤ `nMaxCellsPerSlot` |
| Each slot (dynamic) | `cellPrmStatIdx` in each cell entry | Index back to that cell's static configuration |

## Why limits are counted "per cell group"

The GPU buffers are shared by every cell in the batch. So many cuPHY limits apply to the total across all cells, not to each cell:

| Limit | PDSCH | PUSCH |
|---|---|---|
| Cells | 64 per object (`PDSCH_MAX_CELLS_PER_CELL_GROUP`) | 20 per slot (`MAX_CELLS_PER_SLOT`; 40 with `ENABLE_64C`) |
| UEs / transport blocks | 192 (`PDSCH_MAX_UES_PER_CELL_GROUP`; 256 with `ENABLE_64C`) | 192 (`MAX_N_TBS_PER_CELL_GROUP_SUPPORTED`; 256 with `ENABLE_64C`) |
| UE groups | 128 (`PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP`; 256 with `ENABLE_64C`) | 192 (`MAX_N_USER_GROUPS_SUPPORTED`; 256 with `ENABLE_64C`) |

These constants are defined in `cuPHY/src/cuphy/cuphy.h`.

**Example:** 4 cells scheduling 60 UEs each makes 240 transport blocks. That exceeds the default limit of 192, even though each cell alone is well within it. For PUSCH, exceeding a cell-group limit produces an out-of-range status such as `CUPHY_PUSCH_STATUS_NTBS_PERCELLGROUP_OUT_OF_RANGE`.
