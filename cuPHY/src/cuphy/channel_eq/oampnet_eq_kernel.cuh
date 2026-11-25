/*
 * OAMP-Net2 PUSCH equalizer + soft demapper -- core CUDA device code.
 *
 * CUDA port of utils/oampnet_detector.py (OAMPNet2Detector), shaped for the
 * cuPHY PUSCH receiver (cuPHY/src/cuphy/channel_eq). This header has no cuPHY
 * dependency so it can be unit-tested standalone (see test/); the cuPHY
 * component in oampnet_eq.{hpp,cu} wraps it.
 *
 * Why one fused kernel instead of cuPHY's coefComp + softDemap split:
 *   cuPHY's MMSE-IRC computes W per tone (coefComp) and then applies it to
 *   every data symbol (softDemap). OAMP's linear stage depends on v_t^2, which
 *   is estimated from the residual y - H x_t of *that* RE, and every layer is
 *   followed by a non-linear denoiser, so nothing can be precomputed per tone
 *   except G = Hw^H Hw. The kernel therefore takes (H, LwwInv, Y) and writes
 *   LLRs directly -- it replaces the soft-demap node in the PUSCH graph.
 *
 * Signal model (whitened domain, as in cuPHY MMSE-IRC):
 *   Hw = LwwInv * H_eff, yw = LwwInv * y, R = I, tr(R) = Nr.
 *   White-noise mode (R = sigma^2 I): Hw = H_eff / sigma, yw = y / sigma.
 *   H_eff folds in TDI interpolation and CFO rotation (see loadHeffColumn).
 *
 * Per OAMP-Net2 layer t (paper eqs. 19-28, Python reference line by line):
 *   v2    = max((||yw - Hw x||^2 - Nr) / tr(G), eps)                    (26)
 *   B     = (v2 G + I)^-1                       [Nt x Nt, HPD, eig >= 1]
 *   W     = s B Hw^H,  s = Nt / tr(B G)    (== Nt/tr(W~H) W~, matrix
 *           inversion lemma applied to W~ = v2 Hw^H (v2 Hw Hw^H + I)^-1) (19,20)
 *   r     = x + gamma s B (Hw^H (yw - Hw x))                              (24)
 *   C     = I - theta s B G,   W R W^H = s^2 B G B
 *   tau2  = (tr(C C^H) v2 + theta^2 tr(W W^H)) / Nt        [shared model] (27)
 *   tau2k = [C C^H]_kk v2 + theta_k^2 [W W^H]_kk           [per-stream]
 *   x     = phi (E{x | r, tau2} - xi r)                                (25,28)
 * and the LLRs are the max-log LLRs of the final (r, tau2)                (29).
 *
 * Equivalence with the TF model: identical when R = sigma^2 I (eq. 26 is
 * scale-invariant then). For coloured R, W and tau2 are whitening-invariant
 * but v2 is computed in the whitened domain here vs. the raw domain in TF.
 * The TF model additionally applies a 1e-6 relative diagonal load to the
 * Nr x Nr matrix; here the Nt x Nt matrix v2 G + I has eigenvalues >= 1 and
 * needs none.
 *
 * LLR convention: cuPHY's, i.e. LLR = ln P(b=0)/P(b=1) = -(Sionna LLR), bit
 * order b0..b(Qm-1) as in TS 38.211 5.1 (I carries b0,b2,..; Q carries b1,b3,..).
 */
#if !defined(OAMPNET_EQ_KERNEL_CUH_INCLUDED_)
#define OAMPNET_EQ_KERNEL_CUH_INCLUDED_

#include <cstdint>
#include <cuda_fp16.h>
#if defined(__CUDACC__)
#include <cooperative_groups.h>
#endif

// Host-safe part (constants, descriptors) is visible to host-compiled
// translation units such as cuphy_channels/pusch_rx.cpp (via oampnet_eq.hpp);
// the device code below is compiled only by nvcc.
namespace oampnet_eq {

static constexpr int OAMP_MAX_ITER         = 8; // max unfolded layers T
static constexpr int OAMP_MAX_LAYERS       = 8; // max streams per UE group (cuPHY: CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP)
static constexpr int OAMP_N_TONES_PER_PRB  = 12;
static constexpr int OAMP_MAX_SYMS_PER_SLOT = 14;
static constexpr int OAMP_MAX_PAM_LEVELS   = 16; // 256QAM
static constexpr float OAMP_LLR_LIM        = 65504.0f; // FP16 max, same clamp as cuPHY

//-----------------------------------------------------------------------------
// Static descriptor: trained parameters + numerics. One per processing path
// (full-slot / early-HARQ) because the symbol bitmask lives here.
//-----------------------------------------------------------------------------
struct OampNetStatDescr
{
    // Parameters are always stored per stream; the shared (paper) model is
    // replicated across k by the host loader, so the kernel has one code path.
    float    gamma[OAMP_MAX_ITER][OAMP_MAX_LAYERS];
    float    theta[OAMP_MAX_ITER][OAMP_MAX_LAYERS];
    float    phi  [OAMP_MAX_ITER][OAMP_MAX_LAYERS];
    float    xi   [OAMP_MAX_ITER][OAMP_MAX_LAYERS];
    float    v2Floor;       // eps in max(v^2, eps); TF: 5e-13
    float    tau2Floor;     // eps in max(tau^2, eps); TF: 5e-13
    uint16_t symbolBitmask; // OFDM symbols to process (bit i = symbol i); 0x3FFF = full slot
    uint8_t  numIter;       // T, 1..OAMP_MAX_ITER
    uint8_t  perStream;     // 0: shared tau^2 (eq. 27), 1: per-stream tau^2
};

//-----------------------------------------------------------------------------
// Per-UE-group view of the cuPHY tensors the kernel touches. Strides are in
// elements (complex elements for complex tensors), same as cuphyTensorInfoN_t.
//-----------------------------------------------------------------------------
struct OampNetUeGrpView
{
    const void* pH;   int sH[4];   // H est   (N_BS_ANTS, N_LAYERS, NF, NH)   complex TH
    const void* pLw;  int sLw[3];  // LwwInv  (N_BS_ANTS, N_BS_ANTS, N_PRB)   complex TH
    const void* pY;   int sY[3];   // DataRx  (NF_cell, N_SYMB, N_BS_ANTS)    complex TY
    void*       pLlr; int sLlr[4]; // LLR     (N_LLR, N_LAYERS, NF, ND)       real    TLlr
    const void* pCfo; int sCfo[2]; // CFO est (N_SYMB, N_UE)                  complex TH (optional)

    uint16_t nPrb;
    uint16_t startPrb;   // DataRx is indexed in cell PRBs, H/LLR in UE-group PRBs
    uint8_t  nRxAnt;
    uint8_t  nLayers;
    uint8_t  nDataSym;
    uint8_t  nDmrsSyms;  // total DMRS symbols (all additional positions, x maxLen)
    uint8_t  dmrsMaxLen;
    uint8_t  enableTdi;  // linear time interpolation of H between DMRS instances
    uint8_t  enableCfo;  // apply cuPHY CFO correction (see loadHeffColumn)
    uint8_t  whiteNoise{0};  // 1: R = sigma^2 I, whiten by noiseScale (pLw unused); 0: whiten by LwwInv
    float    noiseScale{1.f}; // 1/sigma = sqrt(1/noise variance), used when whiteNoise == 1

    const uint8_t* dataSymLoc; // [nDataSym]
    const uint8_t* dmrsSymLoc; // [nDmrsSyms]
    const uint8_t* qam;        // [nLayers] bits per symbol (2,4,6,8)
    const uint8_t* layerToUe;  // [nLayers] UE-group layer -> UE index (CFO table column)
};

#if defined(__CUDACC__)
namespace cg = cooperative_groups;

//-----------------------------------------------------------------------------
// Small complex helpers (float2), fp32 compute throughout.
//-----------------------------------------------------------------------------
__device__ __forceinline__ float2 c_add(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 c_sub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 c_scale(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
__device__ __forceinline__ float2 c_mul(float2 a, float2 b) { return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
__device__ __forceinline__ float2 c_conj(float2 a) { return make_float2(a.x, -a.y); }
// a += b * c
__device__ __forceinline__ float2 c_fma(float2 b, float2 c, float2 a)
{
    a.x = fmaf(b.x, c.x, fmaf(-b.y, c.y, a.x));
    a.y = fmaf(b.x, c.y, fmaf(b.y, c.x, a.y));
    return a;
}
// a += conj(b) * c
__device__ __forceinline__ float2 c_fma_cj(float2 b, float2 c, float2 a)
{
    a.x = fmaf(b.x, c.x, fmaf(b.y, c.y, a.x));
    a.y = fmaf(b.x, c.y, fmaf(-b.y, c.x, a.y));
    return a;
}
__device__ __forceinline__ float c_abs2(float2 a) { return fmaf(a.x, a.x, a.y * a.y); }
__device__ __forceinline__ float2 c_inv(float2 a)
{
    const float d = 1.0f / c_abs2(a);
    return make_float2(a.x * d, -a.y * d);
}

template <typename T> struct CplxLoad;
template <> struct CplxLoad<float>
{
    static __device__ __forceinline__ float2 ld(const void* p, int idx)
    {
        return reinterpret_cast<const float2*>(p)[idx];
    }
};
template <> struct CplxLoad<__half>
{
    static __device__ __forceinline__ float2 ld(const void* p, int idx)
    {
        return __half22float2(reinterpret_cast<const __half2*>(p)[idx]);
    }
};

template <typename T> __device__ __forceinline__ void storeLlr(void* p, int idx, float v);
template <> __device__ __forceinline__ void storeLlr<float>(void* p, int idx, float v)
{
    reinterpret_cast<float*>(p)[idx] = v;
}
template <> __device__ __forceinline__ void storeLlr<__half>(void* p, int idx, float v)
{
    reinterpret_cast<__half*>(p)[idx] = __float2half_rz(v);
}

template <int G>
__device__ __forceinline__ float tileSum(const cg::thread_block_tile<G>& tile, float v)
{
#pragma unroll
    for(int o = G / 2; o > 0; o >>= 1) v += tile.shfl_xor(v, o);
    return v;
}

//-----------------------------------------------------------------------------
// One PAM dimension of a 3GPP square QAM (TS 38.211 5.1, Gray labelled).
// Bits c_0..c_{nb-1} of a dimension are QAM bits b_iq, b_{iq+2}, ... .
// amplitude = (1-2c0) * m,  m built from the innermost bit outwards:
//   nb=2: m = 2-(1-2c1); nb=3: m = 4-(1-2c1)(2-(1-2c2)); ...
// scaled to unit average QAM energy (1/sqrt(2), 1/sqrt(10), 1/sqrt(42), 1/sqrt(170)).
//-----------------------------------------------------------------------------
__device__ __forceinline__ float pamLevel(int idx, int nb, float scale)
{
    float m = 1.0f;
    for(int i = nb - 1; i >= 1; --i)
    {
        const float s = 1.0f - 2.0f * static_cast<float>((idx >> i) & 1);
        m             = static_cast<float>(1 << (nb - i)) - s * m;
    }
    return (1.0f - 2.0f * static_cast<float>(idx & 1)) * m * scale;
}

__device__ __forceinline__ float qamScale(int nb)
{
    // 1/sqrt(2(M-1)/3), M = 4^nb
    switch(nb)
    {
    case 1: return 0.70710678118654752f;
    case 2: return 0.31622776601683794f;
    case 3: return 0.15430334996209191f;
    default: return 0.07669649888473705f;
    }
}

// Posterior mean E{a | r, tau2} on one PAM dimension under the AWGN model with
// complex-symbol variance tau2 (logit = -(r-a)^2 / tau2, matching the TF
// model's softmax over -|r-x|^2/tau2, which separates over I and Q).
__device__ __forceinline__ float pamPosteriorMean(float r, float tau2Inv, int nb, float scale)
{
    const int L = 1 << nb;
    float     lmax = -INFINITY;
    for(int i = 0; i < L; ++i)
    {
        const float d = r - pamLevel(i, nb, scale);
        lmax          = fmaxf(lmax, -d * d * tau2Inv);
    }
    float num = 0.f, den = 0.f;
    for(int i = 0; i < L; ++i)
    {
        const float a = pamLevel(i, nb, scale);
        const float d = r - a;
        const float w = __expf(-d * d * tau2Inv - lmax);
        num           = fmaf(w, a, num);
        den += w;
    }
    return num / den;
}

// Max-log LLRs of one PAM dimension, cuPHY sign: (min d^2|c=1 - min d^2|c=0)/tau2.
template <typename TLlr>
__device__ __forceinline__ void pamMaxLogLlr(float r, float tau2Inv, int nb, float scale,
                                             void* pLlr, int llrBase, int llrStride, int iq)
{
    float d0[4], d1[4];
#pragma unroll
    for(int b = 0; b < 4; ++b) { d0[b] = INFINITY; d1[b] = INFINITY; }

    const int L = 1 << nb;
    for(int i = 0; i < L; ++i)
    {
        const float e  = r - pamLevel(i, nb, scale);
        const float d2 = e * e;
#pragma unroll
        for(int b = 0; b < 4; ++b)
        {
            if(b < nb)
            {
                if((i >> b) & 1) d1[b] = fminf(d1[b], d2);
                else             d0[b] = fminf(d0[b], d2);
            }
        }
    }
#pragma unroll
    for(int b = 0; b < 4; ++b)
    {
        if(b < nb)
        {
            float llr = (d1[b] - d0[b]) * tau2Inv;
            llr       = fminf(fmaxf(llr, -OAMP_LLR_LIM), OAMP_LLR_LIM);
            storeLlr<TLlr>(pLlr, llrBase + (2 * b + iq) * llrStride, llr);
        }
    }
}

//-----------------------------------------------------------------------------
// Shared memory layout (float2 elements).
//   block : LwwInv[NR*NR]                          (one PRB)
//   per RE: Hw[NR*NL] (col-major), G[NL*NL] (row-major), yw[NR], res[NR], x[NL], u[NL]
// (the NL x 2NL Gauss-Jordan system lives in registers, one row per thread)
//-----------------------------------------------------------------------------
template <int NR, int NL>
struct OampSmem
{
    static constexpr int PER_RE_RAW = NR * NL + NL * NL + 2 * NR + 2 * NL;
    // odd stride (in float2) so the REs sharing a warp land in different banks
    static constexpr int PER_RE = PER_RE_RAW | 1;
    static constexpr int LW     = NR * NR;
};

// Streams-per-RE tile size NL; symbols per block chosen so the block has
// ~96 threads for every layer count.
template <int NL>
struct OampSymsPerBlk
{
    static constexpr int value = (NL >= 8) ? 1 : (8 / NL);
};

template <int NR, int NL>
__host__ __device__ constexpr int oampSmemBytes()
{
    return (OampSmem<NR, NL>::LW + OAMP_N_TONES_PER_PRB * OampSymsPerBlk<NL>::value * OampSmem<NR, NL>::PER_RE) *
           static_cast<int>(sizeof(float2));
}

template <int NR, int NL>
__host__ __device__ constexpr int oampThreadsPerBlk()
{
    return OAMP_N_TONES_PER_PRB * OampSymsPerBlk<NL>::value * NL;
}

//-----------------------------------------------------------------------------
// Effective channel column k for OFDM symbol symIdx at UE-group tone f.
//   TDI   : linear interpolation between DMRS instances, same index/weight
//           rule as cuPHY eqMmseSoftDemapKernel_v4 (applied to H, not W).
//   CFO   : cuPHY designs W on H_d * cfo(dmrsSym_d) (instance d > 0, coefComp)
//           and rotates the equalised symbol by cfo(sym) (softDemap). With
//           |cfo| = 1 that is the same as detecting on
//           H_eff = H_d * cfo(dmrsSym_d)[d>0] * conj(cfo(sym)).
//-----------------------------------------------------------------------------
template <typename TH, int NR>
__device__ __forceinline__ void loadHeffColumn(const OampNetUeGrpView& v, int k, int f, int symIdx, float2 (&hcol)[NR])
{
    const int nInst = (v.dmrsMaxLen > 0) ? (v.nDmrsSyms / v.dmrsMaxLen) : 1;
    const int L     = v.dmrsMaxLen;

    int   d0 = 0, d1 = 0;
    float a0 = 1.f, a1 = 0.f;
    if(v.enableTdi && nInst > 1)
    {
        int di = 1;
        while((di < nInst - 1) && (v.dmrsSymLoc[(di + 1) * L - 1] < symIdx)) ++di;
        const float t0 = static_cast<float>(v.dmrsSymLoc[(di - 1) * L]);
        const float t1 = static_cast<float>(v.dmrsSymLoc[di * L]);
        a0             = (t1 - static_cast<float>(symIdx)) / (t1 - t0);
        a1             = (static_cast<float>(symIdx) - t0) / (t1 - t0);
        d0             = di - 1;
        d1             = di;
    }

    float2 rot0 = make_float2(1.f, 0.f), rot1 = make_float2(1.f, 0.f);
    if(v.enableCfo && v.pCfo)
    {
        const int    ue   = v.layerToUe[k];
        const float2 cSym = c_conj(CplxLoad<TH>::ld(v.pCfo, symIdx * v.sCfo[0] + ue * v.sCfo[1]));
        rot0              = cSym;
        rot1              = cSym;
        if(d0 > 0) rot0 = c_mul(rot0, CplxLoad<TH>::ld(v.pCfo, v.dmrsSymLoc[d0 * L] * v.sCfo[0] + ue * v.sCfo[1]));
        if(d1 > 0) rot1 = c_mul(rot1, CplxLoad<TH>::ld(v.pCfo, v.dmrsSymLoc[d1 * L] * v.sCfo[0] + ue * v.sCfo[1]));
    }
    rot0 = c_scale(rot0, a0);
    rot1 = c_scale(rot1, a1);

    const int base0 = k * v.sH[1] + f * v.sH[2] + d0 * v.sH[3];
    const int base1 = k * v.sH[1] + f * v.sH[2] + d1 * v.sH[3];
#pragma unroll
    for(int r = 0; r < NR; ++r)
    {
        float2 h = c_mul(rot0, CplxLoad<TH>::ld(v.pH, base0 + r * v.sH[0]));
        if(a1 != 0.f) h = c_fma(rot1, CplxLoad<TH>::ld(v.pH, base1 + r * v.sH[0]), h);
        hcol[r] = h;
    }
}

//-----------------------------------------------------------------------------
// Kernel body. One thread block = one PRB x SPB data symbols; a tile of NL
// threads per RE, thread k owning stream (row) k of every NL x NL matrix.
// blockDim.x = 12 * SPB * NL, gridDim = (nMaxPrb, ceil(nMaxDataSym / SPB), nUeGrps).
// Requires NL == v.nLayers and NR == v.nRxAnt (selected at setup).
//-----------------------------------------------------------------------------
template <typename TH, typename TY, typename TLlr, int NR, int NL>
__device__ __forceinline__ void oampnetEqBody(const OampNetStatDescr& stat,
                                              const OampNetUeGrpView&  v,
                                              uint32_t                 prbIdx,
                                              uint32_t                 symChunkIdx,
                                              float2*                  smem)
{
    static_assert(NL >= 1 && NL <= OAMP_MAX_LAYERS && (NL & (NL - 1)) == 0, "NL must be 1,2,4,8");
    static_assert(NL <= NR, "need at least as many rx antennas as layers");
    constexpr int SPB = OampSymsPerBlk<NL>::value;
    using Smem        = OampSmem<NR, NL>;

    if(prbIdx >= v.nPrb) return; // whole block, before any barrier

    cg::thread_block                 block = cg::this_thread_block();
    cg::thread_block_tile<NL>        tile  = cg::tiled_partition<NL>(block);

    // ---- Whitening. Coloured noise: LwwInv of this PRB, shared by every RE in the
    //      block. White noise (R = sigma^2 I): a scalar 1/sigma, LwwInv not read.
    //      v.whiteNoise is uniform across the block (one UE group per block).
    const bool white = (v.whiteNoise != 0);
    float2*    shLw  = smem; // col-major NR x NR
    if(!white)
    {
        for(int e = threadIdx.x; e < NR * NR; e += blockDim.x)
        {
            const int r = e % NR, c = e / NR;
            shLw[e]     = CplxLoad<TH>::ld(v.pLw, r * v.sLw[0] + c * v.sLw[1] + static_cast<int>(prbIdx) * v.sLw[2]);
        }
    }
    block.sync(); // last block-wide barrier; RE tiles may exit independently below

    const int reInBlk = threadIdx.x / NL;
    const int k       = threadIdx.x % NL;
    const int tone    = reInBlk % OAMP_N_TONES_PER_PRB;
    const int symOff  = reInBlk / OAMP_N_TONES_PER_PRB;
    const int dsi     = static_cast<int>(symChunkIdx) * SPB + symOff; // data-symbol index
    if(dsi >= v.nDataSym) return;
    const int symIdx = v.dataSymLoc[dsi];
    if(!((stat.symbolBitmask >> symIdx) & 1)) return;

    const int f     = static_cast<int>(prbIdx) * OAMP_N_TONES_PER_PRB + tone; // UE-group tone
    const int fCell = f + OAMP_N_TONES_PER_PRB * v.startPrb;                  // cell tone (DataRx)

    float2* shHw  = smem + Smem::LW + reInBlk * Smem::PER_RE; // col-major NR x NL
    float2* shG   = shHw + NR * NL;                           // row-major NL x NL
    float2* shYw  = shG + NL * NL;                            // NR
    float2* shRes = shYw + NR;                                // NR
    float2* shX   = shRes + NR;                               // NL
    float2* shU   = shX + NL;                                 // NL

    // ---- y (raw) -> shRes as scratch, then yw = LwwInv y
    for(int r = k; r < NR; r += NL)
        shRes[r] = CplxLoad<TY>::ld(v.pY, fCell * v.sY[0] + symIdx * v.sY[1] + r * v.sY[2]);

    // ---- Hw(:,k) = LwwInv * H_eff(:,k)   (white noise: H_eff(:,k) / sigma)
    {
        float2 hcol[NR];
        loadHeffColumn<TH, NR>(v, k, f, symIdx, hcol);
#pragma unroll
        for(int r = 0; r < NR; ++r)
        {
            float2 acc;
            if(white)
            {
                acc = c_scale(hcol[r], v.noiseScale);
            }
            else
            {
                acc = make_float2(0.f, 0.f);
#pragma unroll
                for(int j = 0; j < NR; ++j) acc = c_fma(shLw[r + NR * j], hcol[j], acc);
            }
            shHw[r + NR * k] = acc;
        }
    }
    tile.sync();

    // ---- yw = LwwInv * y   (white noise: y / sigma)
    for(int r = k; r < NR; r += NL)
    {
        float2 acc;
        if(white)
        {
            acc = c_scale(shRes[r], v.noiseScale);
        }
        else
        {
            acc = make_float2(0.f, 0.f);
#pragma unroll
            for(int j = 0; j < NR; ++j) acc = c_fma(shLw[r + NR * j], shRes[j], acc);
        }
        shYw[r] = acc;
    }

    // ---- G = Hw^H Hw (row k), tr(G) = ||Hw||_F^2
    // (diagonal picked inside the unrolled loop: indexing a register array
    //  with the runtime lane k would demote it to local memory)
    float2 Gk[NL];
    float  gkk = 0.f;
#pragma unroll
    for(int j = 0; j < NL; ++j)
    {
        float2 acc = make_float2(0.f, 0.f);
#pragma unroll
        for(int r = 0; r < NR; ++r) acc = c_fma_cj(shHw[r + NR * k], shHw[r + NR * j], acc);
        Gk[j]           = acc;
        shG[k * NL + j] = acc;
        if(j == k) gkk = acc.x;
    }
    const float trG = fmaxf(tileSum<NL>(tile, gkk), stat.v2Floor);

    shX[k] = make_float2(0.f, 0.f); // x_1 = 0
    tile.sync();

    const int   nb     = v.qam[k] >> 1;
    const float qScale = qamScale(nb);
    const float nt     = static_cast<float>(NL);
    const int   T      = stat.numIter;

    for(int t = 0; t < T; ++t)
    {
        const float gamma = stat.gamma[t][k];
        const float theta = stat.theta[t][k];

        // (a) residual yw - Hw x and ||.||^2
        float rrPart = 0.f;
        for(int r = k; r < NR; r += NL)
        {
            float2 acc = shYw[r];
#pragma unroll
            for(int j = 0; j < NL; ++j) acc = c_sub(acc, c_mul(shHw[r + NR * j], shX[j]));
            shRes[r] = acc;
            rrPart += c_abs2(acc);
        }
        const float rr = tileSum<NL>(tile, rrPart);

        // (b) eq. 26 in the whitened domain (tr(R) = Nr)
        const float v2 = fmaxf((rr - static_cast<float>(NR)) / trG, stat.v2Floor);

        // (c) [v2 G + I | I] -> Gauss-Jordan -> [I | B]. HPD with eig >= 1, so no
        //     pivoting. Row k lives in thread k's registers; the pivot row is
        //     broadcast with shuffles (no shared memory, no tile barriers).
        float2 aug[2 * NL];
#pragma unroll
        for(int j = 0; j < NL; ++j)
        {
            float2 a = c_scale(Gk[j], v2);
            if(j == k) a.x += 1.f;
            aug[j]      = a;
            aug[NL + j] = make_float2(j == k ? 1.f : 0.f, 0.f);
        }
#pragma unroll
        for(int p = 0; p < NL; ++p)
        {
            if(k == p)
            {
                const float2 piv = c_inv(aug[p]);
#pragma unroll
                for(int c = p; c < 2 * NL; ++c) aug[c] = c_mul(aug[c], piv);
            }
            const float2 fct = aug[p]; // this row's multiplier (unused by the pivot thread)
#pragma unroll
            for(int c = p; c < 2 * NL; ++c)
            {
                const float2 rp = make_float2(tile.shfl(aug[c].x, p), tile.shfl(aug[c].y, p));
                if(k != p) aug[c] = c_sub(aug[c], c_mul(fct, rp));
            }
        }
        float2 Bk[NL]; // row k of B (Hermitian)
#pragma unroll
        for(int j = 0; j < NL; ++j) Bk[j] = aug[NL + j];

        // (d) P = B G (row k); s = Nt / tr(B G)  (W normalisation, eq. 19)
        float2 Pk[NL];
        float  pkk = 0.f;
#pragma unroll
        for(int j = 0; j < NL; ++j)
        {
            float2 acc = make_float2(0.f, 0.f);
#pragma unroll
            for(int m = 0; m < NL; ++m) acc = c_fma(Bk[m], shG[m * NL + j], acc);
            Pk[j] = acc;
            if(j == k) pkk = acc.x;
        }
        const float s = nt / tileSum<NL>(tile, pkk);

        // (e) u = Hw^H (yw - Hw x)
        {
            float2 acc = make_float2(0.f, 0.f);
#pragma unroll
            for(int r = 0; r < NR; ++r) acc = c_fma_cj(shHw[r + NR * k], shRes[r], acc);
            shU[k] = acc;
        }
        tile.sync();

        // (f) eq. 24: r_k = x_k + gamma s (B u)_k
        float2 Bu = make_float2(0.f, 0.f);
#pragma unroll
        for(int m = 0; m < NL; ++m) Bu = c_fma(Bk[m], shU[m], Bu);
        const float2 rk = c_add(shX[k], c_scale(Bu, gamma * s));

        // (g) eq. 27: C = I - theta s P, W W^H = s^2 B G B
        float rowCC = 0.f, wsw = 0.f;
#pragma unroll
        for(int j = 0; j < NL; ++j)
        {
            float2 c = c_scale(Pk[j], -theta * s);
            if(j == k) c.x += 1.f;
            rowCC += c_abs2(c);
            // Re(P_kj conj(B_kj)) = [B G B]_kk summand
            wsw = fmaf(Pk[j].x, Bk[j].x, fmaf(Pk[j].y, Bk[j].y, wsw));
        }
        wsw *= s * s;

        float tau2;
        if(stat.perStream)
        {
            tau2 = rowCC * v2 + theta * theta * wsw;
        }
        else
        {
            // shared model: theta identical across k (replicated by the loader)
            tau2 = (tileSum<NL>(tile, rowCC) * v2 + theta * theta * tileSum<NL>(tile, wsw)) / nt;
        }
        tau2               = fmaxf(tau2, stat.tau2Floor);
        const float t2Inv  = 1.0f / tau2;

        if(t == T - 1)
        {
            // eq. 29: soft output from the final (r_T, tau_T^2)
            const int llrBase = k * v.sLlr[1] + f * v.sLlr[2] + dsi * v.sLlr[3];
            pamMaxLogLlr<TLlr>(rk.x, t2Inv, nb, qScale, v.pLlr, llrBase, v.sLlr[0], 0);
            pamMaxLogLlr<TLlr>(rk.y, t2Inv, nb, qScale, v.pLlr, llrBase, v.sLlr[0], 1);
        }
        else
        {
            // eqs. 25, 28: x_{t+1} = phi (E{x|r,tau2} - xi r)
            const float  phi   = stat.phi[t][k];
            const float  xi    = stat.xi[t][k];
            const float2 xPost = make_float2(pamPosteriorMean(rk.x, t2Inv, nb, qScale),
                                             pamPosteriorMean(rk.y, t2Inv, nb, qScale));
            shX[k]             = c_scale(c_sub(xPost, c_scale(rk, xi)), phi);
            // all reads of shX (steps a, f) precede the barriers above/below
            tile.sync();
        }
    }
}

#endif // defined(__CUDACC__)

} // namespace oampnet_eq

#endif // OAMPNET_EQ_KERNEL_CUH_INCLUDED_
