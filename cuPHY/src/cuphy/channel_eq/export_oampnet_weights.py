#!/usr/bin/env python3
"""Export trained OAMP-Net2 weights for the cuPHY equalizer.

Reads the pickled get_weights() list written by scripts/train_oampnet.py
([gamma, theta, phi, xi], each shape (T,) for the shared model or (T, Nt) for
-per_stream) and writes the binary format read by
puschRxOampNetWeights_t::loadFile():

    char[8]  "OAMPNET2"
    uint32   version (= 1)
    uint32   numIter  T
    uint32   nStreams (1 = shared model)
    float32  gamma[T*nStreams], theta[...], phi[...], xi[...]   (row-major [t][k])

Usage:
    python export_oampnet_weights.py ../weights/cdl_16x8_mu_oampnet_mcs0_ps_weights \
        -o cdl_16x8_mu_oampnet_mcs0_ps.oampnet2
    python export_oampnet_weights.py --classical 4 -o oamp_T4.oampnet2   # untrained OAMP
"""
import argparse
import pickle
import struct

import numpy as np

MAX_ITER, MAX_LAYERS = 8, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", help="pickled OAMP-Net2 weights file")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--classical", type=int, metavar="T",
                    help="write classical-OAMP parameters (gamma=theta=phi=1, xi=0) for T layers")
    args = ap.parse_args()

    if args.classical:
        t = args.classical
        w = [np.ones((t, 1)), np.ones((t, 1)), np.ones((t, 1)), np.zeros((t, 1))]
    else:
        if not args.weights:
            ap.error("give a weights file or --classical T")
        w = pickle.load(open(args.weights, "rb"))
        if len(w) != 4:
            raise SystemExit(f"expected 4 arrays [gamma, theta, phi, xi], got {len(w)}")
        w = [np.asarray(a, np.float32).reshape(np.shape(a)[0], -1) for a in w]

    t, n = w[0].shape
    if not (1 <= t <= MAX_ITER and 1 <= n <= MAX_LAYERS):
        raise SystemExit(f"unsupported shape T={t}, nStreams={n} (max {MAX_ITER}, {MAX_LAYERS})")
    if any(a.shape != (t, n) for a in w):
        raise SystemExit("gamma/theta/phi/xi shapes differ")

    with open(args.out, "wb") as f:
        f.write(b"OAMPNET2")
        f.write(struct.pack("<III", 1, t, n))
        for a in w:
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

    kind = "shared" if n == 1 else f"per-stream ({n})"
    print(f"wrote {args.out}: T={t}, {kind}")
    for name, a in zip(("gamma", "theta", "phi", "xi"), w):
        print(f"  {name:5s} " + np.array2string(a.squeeze(), precision=4, max_line_width=120))


if __name__ == "__main__":
    main()
