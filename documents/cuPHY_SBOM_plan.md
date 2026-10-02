# cuPHY Software BOM: Build Plan and 6G Migration Path

This document has two parts:

1. **Build plan:** how to produce a software bill of materials (SBOM) for cuPHY today, for 5G.
2. **6G migration path:** how that SBOM needs to evolve as SBOM standards, regulations and cuPHY itself move toward 6G.

Source paths are relative to the repository root. "The PDF" means [aerial-cuda-accelerated-ran.pdf](aerial-cuda-accelerated-ran.pdf) in this folder (Aerial CUDA-Accelerated RAN documentation, Release 26.3.0).

---

## Part 1 — Build plan

No single file in the repository lists cuPHY's dependencies with their versions and licenses, so the SBOM has to be assembled from several sources.

### File format

Use **CycloneDX 1.6 JSON** as the main machine-readable SBOM. Export **SPDX 2.3 JSON** from the same data, and add a **spreadsheet** for people to read.

| Format | Why |
|---|---|
| **CycloneDX 1.6 JSON** (`cuphy.cdx.json`) | Widely used standard (ECMA-424). Records versions, package IDs, licenses and the dependency graph. Best supported by vulnerability scanners such as Grype and Dependency-Track. |
| **SPDX 2.3 JSON** (`cuphy.spdx.json`) | ISO standard (ISO/IEC 5962), often requested for license compliance and by customers. Converted from the CycloneDX file, so there is only one source of truth. |
| **Excel** (`cuPHY_SBOM.xlsx`) | A readable view of the same data, matching [cuPHY_5G_NR_Channels_Parameters.xlsx](cuPHY_5G_NR_Channels_Parameters.xlsx). |

If only one file is produced, it should be the CycloneDX JSON.

### What the SBOM covers

| Layer | Contents | Where the data comes from |
|---|---|---|
| **1. cuPHY itself** | `libcuphy`, `cuphy_ldpc`, `cuphy_channels`, `cuphy_hdf5`, `nvlog`, `aerial_utils`, `aerial_common`. Apache-2.0: 998 cuPHY source files carry that license tag. | CMake targets, [aerial-sdk-version](../aerial-sdk-version) |
| **2. Libraries cuPHY links against** | CUDA Toolkit 13.3 (cudart, CUPTI, NVTX, cuRAND), MathDx/cuFFTDx 26.03, TensorRT 10.14.1.48, gsl-lite, yaml-cpp, wise_enum, fmt, CLI11, HDF5 | `find_package` and `target_link_libraries` in [cuPHY/CMakeLists.txt](../cuPHY/CMakeLists.txt) and [cuPHY/src/cuphy/CMakeLists.txt](../cuPHY/src/cuphy/CMakeLists.txt) |
| **3. Prebuilt binaries** | LDPC decoder cubin, version `c3e5b9` | [versions.sh](../cuPHY-CP/container/versions.sh). Where it came from needs to be recorded explicitly. |
| **4. Test-only and optional** | GoogleTest, Google Benchmark, MATLAB, Doxygen | CMake. Marked as not shipped. |
| **5. Build container** | Base image `nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04`, apt and pip packages, DOCA 3.3.0, DPDK 22.11, GDRCopy 2.6, about 41 libraries built from source and pinned to exact commits | [versions.sh](../cuPHY-CP/container/versions.sh), [aerial_base_recipe.py](../cuPHY-CP/container/aerial_base_recipe.py), [aerial_build_devel_recipe.py](../cuPHY-CP/container/aerial_build_devel_recipe.py), [requirements.txt](../cuPHY-CP/container/requirements.txt) |
| **6. Host platform** | GPU driver 610.57.04, kernel, NIC firmware, DOCA OFED | The PDF, Software Manifest (p. 48–49). Recorded as environment information, not as cuPHY components. |

### Steps

**1. Decide the scope.**
- Should the SBOM cover only the cuPHY library (layers 1–4), or the whole container cuPHY ships in (layers 1–6)?
- Should test-only dependencies be listed and flagged, or left out?

These two answers decide how much of the work below is needed.

**2. Read what the repository already declares.** This works on any machine; no build needed.
- Parse the CMake files for direct dependencies.
- Parse `versions.sh` and the container recipes for versions and pinned commits.
- Parse `requirements.txt` for Python packages.

**3. Check what the built binaries actually load.** This has to run inside the Aerial container.
- Build with `cmake --preset minimal-x86` (or `minimal-arm`).
- Run `readelf -d` / `ldd` on `libcuphy.so` and the other cuPHY libraries to list the shared libraries they really load.
- Run `cmake --graphviz` to get the dependency graph between targets.

**4. Scan the container image** with `syft` to catch apt and pip packages and installed binaries. Syft misses header-only libraries built from source into `/usr/local`, such as gsl-lite and wise_enum. Steps 2 and 3 fill that gap.

**5. Merge everything into one CycloneDX file** with a script (`documents/build_sbom.py`, written like [build_xlsx.py](build_xlsx.py)). Each component gets:
- name, version and pinned commit
- a package ID (`pkg:github/…`, `pkg:deb/ubuntu/…`, `pkg:pypi/…`, `pkg:generic/nvidia/…`)
- a standard SPDX license ID and file hashes
- a flag saying whether it ships, is optional, or is test-only
- its dependency links

**6. Check the licenses.** Map each component to its SPDX license ID, then compare against:
- [ATTRIBUTION.rst](../ATTRIBUTION.rst), which lists only a handful of components, and
- the PDF's Acknowledgements (p. 795–856), which list about 36.

Some cuPHY dependencies, such as yaml-cpp, HDF5 and the NVIDIA libraries (CUDA, TensorRT, MathDx), do not appear in either list. Each of those needs checking.

**7. Validate and reconcile.**
- Validate the files with `cyclonedx-cli validate`, and run `grype` or `osv-scanner` for known vulnerabilities.
- Resolve mismatches between sources:
  - The PDF says "Release 26.3.0" while its manifest says 26-2, which matches `versions.sh`.
  - For NVIDIA Container Toolkit, the PDF says "1.17.4 or above" while `versions.sh` pins 1.19.1.

**8. Make it repeatable.** Commit the script and write its outputs to `documents/`. Version each SBOM using `aerial-sdk-version` so it can be regenerated for each release.

### Deliverables in `documents/`

- `build_sbom.py`
- `cuphy.cdx.json`
- `cuphy.spdx.json`
- `cuPHY_SBOM.xlsx`

Steps 2, 5 and 6 can be done on any machine. Steps 3, 4 and part of 7 need the Aerial container on a Linux host.

---

## Part 2 — 6G migration path

"Migration to 6G" affects the SBOM in two ways:

- **Standards and regulations:** the SBOM formats and rules expected to apply to 6G-era network equipment.
- **cuPHY's own content:** what changes in cuPHY as it moves toward 6G, especially AI models becoming part of the physical layer.

### Starting point

- **No 6G code yet.** The only 6G mentions are product messaging: the PDF calls the SDK "AI-native… 5G/6G gNB software" (p. 7), and the [README](../README.md) links to the NVIDIA 6G Developer Program. cuPHY implements 5G NR only.
- **AI models are already in the physical layer.** cuPHY can run a channel estimator as a TensorRT model ([trtengine_chest.cpp](../cuPHY/src/cuphy/ch_est/trtengine_chest.cpp), configured by [chest_trt.yaml](../cuPHY/examples/ch_est/chest_trt.yaml)). pyAerial converts PyTorch/TensorFlow models to ONNX and then to TensorRT (PDF p. 388–389).
- **The 5G SBOM plan does not cover these models.** A TensorRT `.engine` file is a shipped artifact that only works with one TensorRT version and one GPU architecture. The example engine has an MD5 checksum (`2170dd84…`, PDF p. 389), but nothing records how it was trained or what it depends on.

### Phase 1 — 5G baseline (Part 1 of this document)

Produce CycloneDX 1.6 JSON plus SPDX 2.3, built so later phases can extend it rather than redo it:

- **Stable component IDs:** give each component a permanent ID and a package ID, so later SBOMs can refer back to it.
- **Tag each component's 3GPP release:** cuPHY is currently Rel-15/16/17 NR, recorded per component or pipeline.
- **One SBOM per pipeline:** separate SBOMs per channel pipeline (PDSCH, PUSCH, …), composed into one cuPHY SBOM. That way 6G pipelines can be added next to 5G ones later.

### Phase 2 — AI models (already needed today)

- **List models as components:** each ONNX model and TensorRT engine goes in the SBOM. CycloneDX has a machine-learning-model type with a model card; SPDX 3.0 has AI and Dataset profiles.
- **Link each engine to what it depends on:** source model → ONNX version → TensorRT 10.14 → GPU architecture. An engine built for one combination will not load on another, so the SBOM must state the combination.
- **Record training-data origin:** this includes pyAerial / Data Lake datasets and the test vectors from `5GModel`, recorded as datasets.
- **Include the model-selection config:** `chest_trt.yaml` and the channel-estimator selection file (`puschrxChestFactorySettingsFilename` in `cuphyPuschStatPrms_t`) decide which model runs, so they belong in the SBOM.

### Phase 3 — Update the SBOM formats and supply-chain records

Driven by regulation and procurement, not by 6G radio specs.

- **Move to newer formats:** SPDX 3.0 for its AI, Dataset and Build profiles, and the newest CycloneDX version (1.7 or later).
- **Add vulnerability status (VEX) documents:** these state whether a known CVE in a listed component actually affects cuPHY. GPU-heavy stacks get many CVEs in components that are not reachable.
- **Add build provenance and signing:** SLSA-style provenance for the container and the LDPC cubin, plus signed SBOMs (e.g. cosign).
- **Drivers to check:**
  - The **EU Cyber Resilience Act** requires an SBOM in technical documentation. Its reporting obligations start around September 2026 and full obligations around December 2027.
  - **O-RAN Alliance WG11** security specifications include SBOM requirements for O-RAN components.
  - **US CISA** has published minimum SBOM elements.

  The dates and requirements above were written from memory and change over time; check the current versions before relying on them.

### Phase 4 — 6G physical layer

The timeline is uncertain:

- 3GPP Release 20 is the 6G study phase.
- Normative 6G specifications are expected from Release 21, around 2028–2029.
- Commercial 6G is expected around 2030.

Check current 3GPP plans before relying on these dates.

- **New components:** 6G waveforms, channel coding and AI-native receivers will likely be new cuPHY modules. Each becomes a component tagged "6G, Rel-21", alongside the 5G components rather than replacing them, since 5G and 6G will run side by side.
- **AI models become core components:** if 6G standardizes AI-based PHY functions, models move from optional to core. The SBOM then needs model versioning and lifecycle tracking (retraining, model updates in the field) like any other library.
- **Hardware BOM:** firmware and drivers (GPU driver 610.57.04, NIC firmware, kernel; PDF p. 48–49) affect PHY behavior more as functions move to AI. CycloneDX can record hardware and firmware in a linked hardware BOM.
- **Shared-platform composition:** the PDF describes running the RAN and AI inference or dApps on the same GPU (MIG). Each tenant should have its own SBOM, composed into a platform SBOM.

### Recommendation

Do Phase 1 and Phase 2 together now: AI models already ship in cuPHY, and adding them later means retrofitting the SBOM structure. Phase 3 should follow the regulatory deadlines. Phase 4 can wait until 3GPP Release 21 content firms up, as long as Phase 1 keeps per-pipeline SBOMs and release tags.
