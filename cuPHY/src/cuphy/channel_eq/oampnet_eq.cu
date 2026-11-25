/*
 * OAMP-Net2 PUSCH equalizer component for cuPHY -- kernels + setup.
 * See oampnet_eq.hpp for the integration contract.
 */
#include "oampnet_eq.hpp"
#include "cuphy.h"
#include "cuphy_internal.h"
#include <cstdio>
#include <cstring>

namespace channel_eq
{
using namespace oampnet_eq;

//-----------------------------------------------------------------------------
// cuPHY kernel: same signature shape as eqMmseSoftDemapKernel, so it can be
// swapped into the soft-demap graph node.
// grid  = (nMaxPrb, ceil(nMaxDataSym / SPB), nUeGrps in het cfg)
// block = 12 * SPB * NL threads, dynamic smem = oampSmemBytes<NR, NL>()
//-----------------------------------------------------------------------------
template <typename TH, typename TY, typename TLlr, int NR, int NL>
__global__ void __launch_bounds__(oampThreadsPerBlk<NR, NL>())
oampnetEqSoftDemapKernel(const puschRxOampNetStatDescr_t* pStatDescr,
                         const __grid_constant__ puschRxChEqSoftDemapDynDescr_t dynDescr)
{
    extern __shared__ float2 smem[];

    const uint32_t                 ueGrpIdx = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    cuphyPuschRxUeGrpPrms_t const& g        = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];

    OampNetUeGrpView v;
    v.pH = g.tInfoHEst.pAddr;
    for(int i = 0; i < 4; ++i) v.sH[i] = g.tInfoHEst.strides[i];
    v.pLw = g.tInfoLwInv.pAddr;
    for(int i = 0; i < 3; ++i) v.sLw[i] = g.tInfoLwInv.strides[i];
    v.pY = g.tInfoDataRx.pAddr;
    for(int i = 0; i < 3; ++i) v.sY[i] = g.tInfoDataRx.strides[i];
    v.pLlr = g.tInfoLLR.pAddr;
    for(int i = 0; i < 4; ++i) v.sLlr[i] = g.tInfoLLR.strides[i];
    v.pCfo    = g.tInfoCfoEst.pAddr;
    v.sCfo[0] = g.tInfoCfoEst.strides[0];
    v.sCfo[1] = g.tInfoCfoEst.strides[1];

    v.nPrb       = g.nPrb;
    v.startPrb   = g.startPrb;
    v.nRxAnt     = static_cast<uint8_t>(g.nRxAnt);
    v.nLayers    = static_cast<uint8_t>(g.nLayers);
    v.nDataSym   = g.nDataSym;
    v.nDmrsSyms  = static_cast<uint8_t>(g.nDmrsSyms);
    v.dmrsMaxLen = g.dmrsMaxLen;
    // Early-HARQ (sub-slot) path: only the front-loaded DMRS estimate exists when
    // this kernel runs, so H of later DMRS instances and the CFO estimate (needs
    // >= 2 DMRS instances) must not be read. Same rule as the MMSE soft demapper.
    const bool fullSlot = (pStatDescr->symbolBitmask == CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK);
    v.enableTdi  = fullSlot ? static_cast<uint8_t>(g.enablePuschTdi) : static_cast<uint8_t>(0);
    v.enableCfo  = fullSlot ? static_cast<uint8_t>(g.enableCfoCorrection) : static_cast<uint8_t>(0);
    // Whitening mode from the derived eqCoeffAlgo (PuschRx maps OAMPNET2 ->
    // MMSE_IRC_SHRINK_OAS and OAMPNET2_WHITE -> NOISE_DIAG_MMSE).
    v.whiteNoise = (g.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    v.noiseScale = sqrtf(fmaxf(g.invNoiseVarLin, 0.f)); // written on GPU by the noise/intf estimator
    v.dataSymLoc = g.dataSymLoc;
    v.dmrsSymLoc = g.dmrsSymLoc;
    v.qam        = g.qam;
    v.layerToUe  = g.ueGrpLayerToUeIdx;

    oampnetEqBody<TH, TY, TLlr, NR, NL>(*pStatDescr, v, blockIdx.x, blockIdx.y, smem);
}

//-----------------------------------------------------------------------------
// Weights
//-----------------------------------------------------------------------------
puschRxOampNetWeights_t puschRxOampNetWeights_t::classicalOamp(uint8_t T)
{
    puschRxOampNetWeights_t w;
    w.numIter  = T;
    w.nStreams = 1;
    for(int t = 0; t < T; ++t)
    {
        w.gamma[t] = 1.f;
        w.theta[t] = 1.f;
        w.phi[t]   = 1.f;
        w.xi[t]    = 0.f;
    }
    return w;
}

// File format (little endian), written by export_oampnet_weights.py:
//   char[8] "OAMPNET2", uint32 version (=1), uint32 numIter, uint32 nStreams,
//   float32 gamma[numIter*nStreams], theta[...], phi[...], xi[...]
cuphyStatus_t puschRxOampNetWeights_t::loadFile(const char* path, puschRxOampNetWeights_t& w)
{
    FILE* fp = std::fopen(path, "rb");
    if(!fp) return CUPHY_STATUS_INVALID_ARGUMENT;
    char     magic[8];
    uint32_t hdr[3];
    bool     ok = (std::fread(magic, 1, 8, fp) == 8) && (std::memcmp(magic, "OAMPNET2", 8) == 0) &&
              (std::fread(hdr, sizeof(uint32_t), 3, fp) == 3) && (hdr[0] == 1) && (hdr[1] >= 1) &&
              (hdr[1] <= OAMP_MAX_ITER) && (hdr[2] >= 1) && (hdr[2] <= OAMP_MAX_LAYERS);
    if(ok)
    {
        w.numIter        = static_cast<uint8_t>(hdr[1]);
        w.nStreams       = static_cast<uint8_t>(hdr[2]);
        const size_t n   = static_cast<size_t>(hdr[1]) * hdr[2];
        float*       dst[4] = {w.gamma, w.theta, w.phi, w.xi};
        for(float* d : dst) ok = ok && (std::fread(d, sizeof(float), n, fp) == n);
    }
    std::fclose(fp);
    return ok ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INVALID_ARGUMENT;
}

void puschRxOampNetEq::fillStatDescr(puschRxOampNetStatDescr_t& d, puschRxOampNetWeights_t const& w, uint16_t symbolBitmask) const
{
    std::memset(&d, 0, sizeof(d));
    for(int t = 0; t < w.numIter; ++t)
    {
        for(int k = 0; k < OAMP_MAX_LAYERS; ++k)
        {
            // shared model -> replicate the scalar; per-stream -> stream k (clamped)
            const int src = t * w.nStreams + ((w.nStreams == 1) ? 0 : std::min<int>(k, w.nStreams - 1));
            d.gamma[t][k] = w.gamma[src];
            d.theta[t][k] = w.theta[src];
            d.phi[t][k]   = w.phi[src];
            d.xi[t][k]    = w.xi[src];
        }
    }
    d.v2Floor       = 5e-13f;
    d.tau2Floor     = 5e-13f;
    d.symbolBitmask = symbolBitmask;
    d.numIter       = w.numIter;
    d.perStream     = (w.nStreams > 1) ? 1 : 0;
}

void puschRxOampNetEq::getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes)
{
    statDescrSizeBytes  = sizeof(puschRxOampNetStatDescr_t);
    statDescrAlignBytes = alignof(puschRxOampNetStatDescr_t);
}

// Static descriptor index <-> soft-demap symbol bitmask
static constexpr uint16_t kStatBitmasks[puschRxOampNetEq::N_STAT_DESCRS] = {
    CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK,
    CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK,
    CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK_MMIMO_EXTRA_DMRS};

int puschRxOampNetEq::statIdxFromBitmask(uint16_t symbolBitmask)
{
    for(int i = 0; i < N_STAT_DESCRS; ++i)
        if(kStatBitmasks[i] == symbolBitmask) return i;
    return -1;
}

int puschRxOampNetEq::pathFromBitmask(uint16_t symbolBitmask)
{
    return (symbolBitmask == CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? CUPHY_PUSCH_FULL_SLOT_PATH : CUPHY_PUSCH_SUB_SLOT_PATH;
}

cuphyStatus_t puschRxOampNetEq::init(puschRxOampNetWeights_t const& weights,
                                     bool                           enableCpuToGpuDescrAsyncCpy,
                                     void**                         ppStatDescrCpu,
                                     void**                         ppStatDescrGpu,
                                     cudaStream_t                   strm)
{
    if(!ppStatDescrCpu || !ppStatDescrGpu) return CUPHY_STATUS_INVALID_ARGUMENT;
    for(int p = 0; p < N_STAT_DESCRS; ++p)
    {
        if(!ppStatDescrCpu[p] || !ppStatDescrGpu[p]) return CUPHY_STATUS_INVALID_ARGUMENT;
        m_statCpu[p] = static_cast<puschRxOampNetStatDescr_t*>(ppStatDescrCpu[p]);
        m_statGpu[p] = static_cast<puschRxOampNetStatDescr_t*>(ppStatDescrGpu[p]);
    }
    return setWeights(weights, enableCpuToGpuDescrAsyncCpy, strm);
}

cuphyStatus_t puschRxOampNetEq::setWeights(puschRxOampNetWeights_t const& weights, bool enableCpuToGpuDescrAsyncCpy, cudaStream_t strm)
{
    if(weights.numIter < 1 || weights.numIter > OAMP_MAX_ITER || weights.nStreams < 1 || weights.nStreams > OAMP_MAX_LAYERS)
        return CUPHY_STATUS_INVALID_ARGUMENT;
    m_weights = weights;
    for(int p = 0; p < N_STAT_DESCRS; ++p)
    {
        if(!m_statCpu[p]) return CUPHY_STATUS_INVALID_ARGUMENT;
        fillStatDescr(*m_statCpu[p], m_weights, kStatBitmasks[p]);
        if(enableCpuToGpuDescrAsyncCpy)
        {
            CUDA_CHECK(cudaMemcpyAsync(m_statGpu[p], m_statCpu[p], sizeof(puschRxOampNetStatDescr_t), cudaMemcpyHostToDevice, strm));
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

bool puschRxOampNetEq::isSupported(cuphyPuschRxUeGrpPrms_t const* pDrvdUeGrpPrmsCpu, uint16_t nUeGrps)
{
    for(uint16_t i = 0; i < nUeGrps; ++i)
    {
        cuphyPuschRxUeGrpPrms_t const& g = pDrvdUeGrpPrmsCpu[i];
        const bool antOk   = (g.nRxAnt == 2) || (g.nRxAnt == 4) || (g.nRxAnt == 8) || (g.nRxAnt == 16);
        const bool layerOk = (g.nLayers == 1) || (g.nLayers == 2) || (g.nLayers == 4) || (g.nLayers == 8);
        if(!antOk || !layerOk || g.nLayers > g.nRxAnt) return false;
        if(g.enableTfPrcd || g.nDmrsCdmGrpsNoData == 1) return false;
        // Needs a noise model from the noise/interference estimator:
        //  - MMSE_IRC*: full Rww -> LwwInv (coloured-noise whitening);
        //  - NOISE_DIAG_MMSE: scalar invNoiseVarLin (white-noise scaling; LwwInv
        //    is not written in this mode and is not read).
        // RZF provides neither. PuschRx maps OAMPNET2 -> MMSE_IRC_SHRINK_OAS and
        // OAMPNET2_WHITE -> NOISE_DIAG_MMSE in the derived UE-group params.
        const bool noiseModelOk = (g.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC) ||
                                  (g.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW) ||
                                  (g.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS) ||
                                  (g.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE);
        if(!noiseModelOk) return false;
        for(uint32_t l = 0; l < g.nLayers; ++l)
            if(g.qam[l] < 2 || g.qam[l] > 8 || (g.qam[l] & 1)) return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
// Kernel selection: (nRxAnt, nLayers) x data types
//-----------------------------------------------------------------------------
template <typename TH, typename TY, typename TLlr, int NR, int NL>
static cuphyStatus_t selectNrNl(CUfunction& func, int& smemBytes, int& nThreads, int& spb)
{
    if constexpr(NL > NR)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    else
    {
        void* k   = reinterpret_cast<void*>(oampnetEqSoftDemapKernel<TH, TY, TLlr, NR, NL>);
        smemBytes = oampSmemBytes<NR, NL>();
        nThreads  = oampThreadsPerBlk<NR, NL>();
        spb       = OampSymsPerBlk<NL>::value;
        {
            MemtraceDisableScope md;
            if(smemBytes > 48 * 1024)
                CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
            CUDA_CHECK(cudaGetFuncBySymbol(&func, k));
        }
        return CUPHY_STATUS_SUCCESS;
    }
}

template <typename TH, typename TY, typename TLlr, int NR>
static cuphyStatus_t selectNl(uint16_t nLayers, CUfunction& func, int& smemBytes, int& nThreads, int& spb)
{
    switch(nLayers)
    {
    case 1: return selectNrNl<TH, TY, TLlr, NR, 1>(func, smemBytes, nThreads, spb);
    case 2: return selectNrNl<TH, TY, TLlr, NR, 2>(func, smemBytes, nThreads, spb);
    case 4: return selectNrNl<TH, TY, TLlr, NR, 4>(func, smemBytes, nThreads, spb);
    case 8: return selectNrNl<TH, TY, TLlr, NR, 8>(func, smemBytes, nThreads, spb);
    default: return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
}

template <typename TH, typename TY, typename TLlr>
static cuphyStatus_t selectNr(uint16_t nRxAnt, uint16_t nLayers, CUfunction& func, int& smemBytes, int& nThreads, int& spb)
{
    switch(nRxAnt)
    {
    case 2: return selectNl<TH, TY, TLlr, 2>(nLayers, func, smemBytes, nThreads, spb);
    case 4: return selectNl<TH, TY, TLlr, 4>(nLayers, func, smemBytes, nThreads, spb);
    case 8: return selectNl<TH, TY, TLlr, 8>(nLayers, func, smemBytes, nThreads, spb);
    case 16: return selectNl<TH, TY, TLlr, 16>(nLayers, func, smemBytes, nThreads, spb);
    default: return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
}

cuphyStatus_t puschRxOampNetEq::kernelSelect(uint16_t nRxAnt, uint16_t nLayers, cuphyDataType_t hType, cuphyDataType_t dataRxType,
                                             cuphyDataType_t llrType, CUfunction& func, int& smemBytes, int& nThreads, int& spb)
{
    // Same type combos the MMSE soft demapper builds: H/LwwInv fp32, DataRx fp16 (or fp32 under ASIM), LLR fp16.
    if(CUPHY_C_32F == hType && CUPHY_C_16F == dataRxType && CUPHY_R_16F == llrType)
        return selectNr<float, __half, __half>(nRxAnt, nLayers, func, smemBytes, nThreads, spb);
#ifdef ASIM_CUPHY_FP32
    if(CUPHY_C_32F == hType && CUPHY_C_32F == dataRxType && CUPHY_R_16F == llrType)
        return selectNr<float, float, __half>(nRxAnt, nLayers, func, smemBytes, nThreads, spb);
#endif
    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: no OAMP-Net2 kernel for requested data types", __FUNCTION__);
    return CUPHY_STATUS_UNSUPPORTED_TYPE;
}

//-----------------------------------------------------------------------------
// Setup: batch UE groups by (nRxAnt, nLayers) into heterogeneous configs,
// one launch config per het cfg (mirrors puschRxChEq::batchEqSoftDemap).
//-----------------------------------------------------------------------------
cuphyStatus_t puschRxOampNetEq::setup(cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                      cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                      uint16_t                      nUeGrps,
                                      uint16_t                      symbolBitmask,
                                      cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs)
{
    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    if(!isSupported(pDrvdUeGrpPrmsCpu, nUeGrps)) return CUPHY_STATUS_UNSUPPORTED_TYPE;

    const int statIdx = statIdxFromBitmask(symbolBitmask);
    if(statIdx < 0 || !m_statGpu[statIdx]) return CUPHY_STATUS_INVALID_ARGUMENT;
    const int path = pathFromBitmask(symbolBitmask);

    auto& hetCfgs = m_hetCfgs[path];
    auto& dyn     = m_dynDescrCpu[path];
    uint32_t nHet = 0;

    for(uint16_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        cuphyPuschRxUeGrpPrms_t const& g = pDrvdUeGrpPrmsCpu[ueGrpIdx];
        uint32_t                       h = 0;
        for(; h < nHet; ++h)
            if(hetCfgs[h].nRxAnt == g.nRxAnt && hetCfgs[h].nLayers == g.nLayers) break;

        if(h == nHet)
        {
            if(nHet >= CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: exceeded heterogeneous config limit", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }
            hetCfgs[h] = HetCfg{nullptr, g.nRxAnt, g.nLayers, 0, 0, 0};
            ++nHet;
        }
        HetCfg& hc = hetCfgs[h];
        if(hc.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED) return CUPHY_STATUS_INTERNAL_ERROR;
        hc.nMaxPrb                         = std::max<uint16_t>(hc.nMaxPrb, g.nPrb);
        hc.nMaxDataSym                     = std::max<uint16_t>(hc.nMaxDataSym, g.nDataSym);
        dyn[h].hetCfgUeGrpMap[hc.nUeGrps++] = ueGrpIdx;
    }

    // Select every kernel before touching *pLaunchCfgs, so a failure leaves the
    // caller's (MMSE) launch configs intact and usable as a fallback.
    int smemArr[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS]{};
    int nThrArr[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS]{};
    int spbArr[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS]{};
    for(uint32_t h = 0; h < nHet; ++h)
    {
        HetCfg&                        hc = hetCfgs[h];
        cuphyPuschRxUeGrpPrms_t const& g0 = pDrvdUeGrpPrmsCpu[dyn[h].hetCfgUeGrpMap[0]];
        cuphyStatus_t st = kernelSelect(hc.nRxAnt, hc.nLayers, g0.tInfoHEst.elemType, g0.tInfoDataRx.elemType,
                                        g0.tInfoLLR.elemType, hc.func, smemArr[h], nThrArr[h], spbArr[h]);
        if(CUPHY_STATUS_SUCCESS != st) return st;
    }

    cuphyPuschRxChEqLaunchCfgs_t& launchCfgs = *pLaunchCfgs;
    launchCfgs.nCfgs                         = nHet;
    for(uint32_t h = 0; h < nHet; ++h)
    {
        HetCfg const& hc  = hetCfgs[h];
        const int     spb = spbArr[h];

        dyn[h].pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;
        m_statArg[path][h]    = m_statGpu[statIdx];

        cuphyPuschRxChEqLaunchCfg_t& lc = launchCfgs.cfgs[h];
        CUDA_KERNEL_NODE_PARAMS&     kp = lc.kernelNodeParamsDriver;
        kp.func                         = hc.func;
        kp.gridDimX                     = hc.nMaxPrb;
        kp.gridDimY                     = (hc.nMaxDataSym + spb - 1) / spb;
        kp.gridDimZ                     = hc.nUeGrps;
        kp.blockDimX                    = static_cast<unsigned int>(nThrArr[h]);
        kp.blockDimY                    = 1;
        kp.blockDimZ                    = 1;
        kp.sharedMemBytes               = static_cast<unsigned int>(smemArr[h]);
        kp.extra                        = nullptr;

        lc.kernelArgs[0] = &m_statArg[path][h];
        lc.kernelArgs[1] = &dyn[h]; // grid-constant: copied by value into the node
        kp.kernelParams  = &lc.kernelArgs[0];
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace channel_eq
