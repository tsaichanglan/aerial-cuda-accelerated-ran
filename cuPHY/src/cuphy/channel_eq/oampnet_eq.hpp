/*
 * OAMP-Net2 PUSCH equalizer component for cuPHY.
 *
 * Drop into cuPHY/src/cuphy/channel_eq/ next to channel_eq.{hpp,cu}.
 *
 * Replaces the soft-demap stage of the PUSCH graph (coefComp may stay enabled:
 * its ReeDiagInv still feeds post-EQ SINR / noise reporting). The kernel uses
 * the *same* argument signature as eqMmseSoftDemapKernel:
 *
 *     kernel(StatDescr* pStat, const __grid_constant__ puschRxChEqSoftDemapDynDescr_t dyn)
 *
 * so the launch configs it produces can be written straight into
 * PuschRx::m_chEqSoftDemapLaunchCfgs[path] and pushed to the existing
 * soft-demap graph nodes by cuphy_utils::setHetCfgNodeStatus() (which uses
 * cuGraphExecKernelNodeSetParams and so may swap the kernel function).
 *
 * Inputs read per UE group: tInfoHEst, tInfoLwInv, tInfoDataRx, tInfoCfoEst
 * (if enableCfoCorrection), dmrs/data symbol tables, qam[]. Output: tInfoLLR
 * in the soft demapper's layout and sign convention.
 *
 * Selected by eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_OAMPNET2 or _OAMPNET2_WHITE.
 * PuschRx swaps only the soft-demap launch configs and runs the upstream stages as
 *  - OAMPNET2       -> MMSE_IRC_SHRINK_OAS: full Rww, OAS shrinkage, Cholesky ->
 *                      LwwInv; the kernel whitens with LwwInv (coloured noise).
 *  - OAMPNET2_WHITE -> NOISE_DIAG_MMSE: scalar invNoiseVarLin; the kernel scales
 *                      H and y by sqrt(invNoiseVarLin) (R = sigma^2 I, as trained).
 * The kernel picks the mode per UE group from the derived eqCoeffAlgo. RZF
 * provides no noise model and is rejected by isSupported().
 *
 * Time handling: H is linearly interpolated between DMRS instances per data
 * symbol (TDI) *before* detection, i.e. chEst -> interp(H) -> OAMP-Net2, rather
 * than interpolating equalizer coefficients as the MMSE path does. As in the
 * MMSE soft demapper, TDI and CFO correction are disabled on the early-HARQ
 * (sub-slot) path, where only the front-loaded DMRS estimate exists.
 *
 * Not supported (setup returns CUPHY_STATUS_UNSUPPORTED_TYPE and the caller
 * should keep the MMSE soft demapper): DFT-s-OFDM / transform precoding,
 * data on DMRS symbols (nDmrsCdmGrpsNoData == 1, tInfoLLRCdm1), nLayers > 8,
 * O-RAN 7.2e split, nRxAnt not in {2,4,8,16}. tInfoDataEq is not written.
 */
#if !defined(OAMPNET_EQ_HPP_INCLUDED_)
#define OAMPNET_EQ_HPP_INCLUDED_

#include "channel_eq/channel_eq.hpp"
#include "oampnet_eq_kernel.cuh"
#include <array>

namespace channel_eq
{

using puschRxOampNetStatDescr_t = oampnet_eq::OampNetStatDescr;

// Host-side parameter set, as exported by cuphy_oampnet/export_oampnet_weights.py.
struct puschRxOampNetWeights_t
{
    uint8_t numIter{4};   // T
    uint8_t nStreams{1};  // 1 = shared (paper) model; else per-stream width (<= 8)
    // [numIter][nStreams], row-major
    float gamma[oampnet_eq::OAMP_MAX_ITER * oampnet_eq::OAMP_MAX_LAYERS]{};
    float theta[oampnet_eq::OAMP_MAX_ITER * oampnet_eq::OAMP_MAX_LAYERS]{};
    float phi  [oampnet_eq::OAMP_MAX_ITER * oampnet_eq::OAMP_MAX_LAYERS]{};
    float xi   [oampnet_eq::OAMP_MAX_ITER * oampnet_eq::OAMP_MAX_LAYERS]{};

    // gamma=theta=phi=1, xi=0: untrained OAMP-Net2 == classical OAMP
    static puschRxOampNetWeights_t classicalOamp(uint8_t T);
    // Binary file written by export_oampnet_weights.py (see format there)
    static cuphyStatus_t loadFile(const char* path, puschRxOampNetWeights_t& w);
};

class puschRxOampNetEq
{
public:
    // One static descriptor per soft-demap symbol bitmask: full slot, early-HARQ,
    // early-HARQ with mMIMO extra DMRS. The bitmask can change per slot (it depends
    // on the slot's max DMRS length), so setup() selects a descriptor by pointer
    // instead of rewriting one with a blocking copy.
    static constexpr int N_STAT_DESCRS = 3;

    puschRxOampNetEq()                                   = default;
    puschRxOampNetEq(puschRxOampNetEq const&)            = delete;
    puschRxOampNetEq& operator=(puschRxOampNetEq const&) = delete;

    static void getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes);

    // ppStatDescrCpu/Gpu: N_STAT_DESCRS descriptors each (pinned host / device),
    // allocated by the caller from getDescrInfo() like the other components.
    cuphyStatus_t init(puschRxOampNetWeights_t const& weights,
                       bool                           enableCpuToGpuDescrAsyncCpy,
                       void**                         ppStatDescrCpu,
                       void**                         ppStatDescrGpu,
                       cudaStream_t                   strm);

    // Reload trained parameters (e.g. per-MCS weight sets) between slots.
    cuphyStatus_t setWeights(puschRxOampNetWeights_t const& weights, bool enableCpuToGpuDescrAsyncCpy, cudaStream_t strm);

    // True if every UE group can be handled by this equalizer.
    static bool isSupported(cuphyPuschRxUeGrpPrms_t const* pDrvdUeGrpPrmsCpu, uint16_t nUeGrps);

    // Mirrors puschRxChEq::setupSoftDemap. symbolBitmask selects the path:
    // CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK -> full slot,
    // CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK[_MMIMO_EXTRA_DMRS] -> sub-slot.
    // pLaunchCfgs is overwritten.
    cuphyStatus_t setup(cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                        cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                        uint16_t                      nUeGrps,
                        uint16_t                      symbolBitmask,
                        cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs);

private:
    static int statIdxFromBitmask(uint16_t symbolBitmask);
    static int pathFromBitmask(uint16_t symbolBitmask);
    void fillStatDescr(puschRxOampNetStatDescr_t& d, puschRxOampNetWeights_t const& w, uint16_t symbolBitmask) const;
    cuphyStatus_t kernelSelect(uint16_t nRxAnt, uint16_t nLayers, cuphyDataType_t hType, cuphyDataType_t dataRxType,
                               cuphyDataType_t llrType, CUfunction& func, int& smemBytes, int& nThreads, int& symsPerBlk);

    struct HetCfg
    {
        CUfunction func;
        uint16_t   nRxAnt;
        uint16_t   nLayers;
        uint16_t   nMaxPrb;
        uint16_t   nMaxDataSym;
        uint16_t   nUeGrps;
    };

    puschRxOampNetWeights_t m_weights{};
    static constexpr int N_PATHS = 2; // CUPHY_PUSCH_FULL_SLOT_PATH / CUPHY_PUSCH_SUB_SLOT_PATH

    std::array<puschRxOampNetStatDescr_t*, N_STAT_DESCRS> m_statCpu{};
    std::array<puschRxOampNetStatDescr_t*, N_STAT_DESCRS> m_statGpu{};

    // Per path: kernel args must outlive setup (node params point at them).
    std::array<puschRxChEqSoftDemapDynDescrVec_t, N_PATHS> m_dynDescrCpu{};
    std::array<std::array<puschRxOampNetStatDescr_t*, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>, N_PATHS> m_statArg{};
    std::array<std::array<HetCfg, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>, N_PATHS> m_hetCfgs{};
};

} // namespace channel_eq

#endif // OAMPNET_EQ_HPP_INCLUDED_
