#!/usr/bin/env python3
"""Learn the regime selector: partition the block-feature grid into N regimes, minimising held-out
cross-entropy under the ENGINE 6 smoothing hierarchy (own ctx34 -> c14 -> regime -> cr x vk prior).

  learn_reg.py blockstats.bin gridid [--regs N] [--K list] [--ref b0.bin] [--out model.json]
"""
import argparse, json
import numpy as np

NCTX, NSC, NGRP = 34, 228, 2
DIMS = {0: [8], 1: [8, 8, 8], 2: [8, 8, 8], 3: [8, 8, 8], 4: [8, 8, 8], 5: [8, 8, 8], 6: [8, 8, 8], 7: [8, 8, 8], 8: [8, 8, 8], 9: [8, 8, 8], 10: [8, 8, 8], 11: [8, 8, 8], 12: [8, 8, 8], 13: [8, 8, 8]}
GN = {0: "engine6 hand-tuned", 1: "bw(sum_rl),hi_vk,sum_ctx", 2: "mean_vb,density,bw(sum_rl)",
      3: "mean_vb,max_vb,density", 4: "bw(sum_rl),mean_vb,mean_cr", 5: "sum_ctx,hi_vk,mean_vb",
      6: "mean_vb,vk1_rate,bw(sum_rl)", 7: "bw(sum_rl),density,sum_ctx",
      8: "EXACT rl,hi,ctx", 9: "EXACT rl,hi,mean_vb", 10: "EXACT rl,ctx,mean_vb",
      11: "mean_vb,density,EXACT rl", 12: "EXACT ctx,hi,mean_vb", 13: "EXACT rl,mean_vb,density"}


def _c14(c):
    if c == 0:
        return 0
    vk = (c - 1) // 3 + 1
    cr = (c - 1) % 3
    kg = vk if vk <= 3 else (4 if vk <= 5 else (5 if vk <= 8 else 6))
    return kg * 2 + (1 if cr > 0 else 0)


C14 = np.array([_c14(c) for c in range(NCTX)], dtype=np.int32)


def load(path, gid):
    raw = np.fromfile(path, dtype=np.uint32)
    h = raw[:6]
    assert int(h[0]) == 0x374B4C42
    nfold, ncell, nctx, nsymc, ngrp = (int(x) for x in h[1:6])
    dims = DIMS[gid]
    assert int(np.prod(dims)) == ncell
    H = raw[6:].reshape(nfold, ngrp, *dims, nctx, nsymc)
    return H, dims


def norm(x):
    return x / np.maximum(x.sum(axis=-1, keepdims=True), 1.0)


def prior(L1):
    cr = L1.reshape(19, 12).sum(axis=1); cr = (cr + 1.0) / (cr.sum() + 19)
    vk = L1.reshape(19, 12).sum(axis=0); vk = (vk + 1.0) / (vk.sum() + 12)
    return np.outer(cr, vk).ravel()


class RegCost:
    """bits(H_train, H_test) for one regime, shape (grp, ctx, sym)"""

    def __init__(self, K):
        self.K = K

    def __call__(self, a, e):
        K = self.K
        tot = 0.0
        for g in range(a.shape[0]):
            ag = a[g]; eg = e[g]
            L1 = ag.sum(axis=0)
            p1 = norm(L1); pf = prior(L1)
            a2 = np.zeros((14, NSC))
            for c in range(NCTX):
                a2[C14[c]] += ag[c]
            p2 = norm(a2); n2 = a2.sum(axis=-1)
            n3 = ag.sum(axis=-1)[..., None]; p3 = norm(ag)
            A = n3 / (n3 + K)
            Cc = (1.0 - A) * (n2[C14][..., None] / (n2[C14][..., None] + K * 0.25))
            D = (1.0 - A - Cc) * 0.80
            F = 1.0 - A - Cc - D
            p = A * p3 + Cc * p2[C14] + D * p1 + F * pf
            tot += float(-(eg * np.log2(np.maximum(p, 1e-30))).sum())
        return tot


def sl_of(bounds):
    return tuple(slice(lo, hi) for lo, hi in bounds)


class Grower:
    def __init__(self, H, dims, K, lf, ef):
        self.H, self.dims, self.lf, self.ef = H, dims, lf, ef
        self.nd = len(dims)
        self.cost = RegCost(K)

    def marg(self, bounds, fold):
        A = self.H[(fold, slice(None)) + sl_of(bounds)]
        ax = tuple(range(1, 1 + self.nd))
        return A.sum(axis=ax, dtype=np.float64)

    def along(self, bounds, fold, k):
        A = self.H[(fold, slice(None)) + sl_of(bounds)]
        ax = tuple(1 + j for j in range(self.nd) if j != k)
        return A.sum(axis=ax, dtype=np.float64)

    def grow(self, nregs, verbose=False):
        root = tuple((0, d) for d in self.dims)
        nodes = [root]
        cur = [(self.marg(root, self.lf), self.marg(root, self.ef))]
        total = self.cost(*cur[0])
        curve = [(1, total)]
        while len(nodes) < nregs:
            best = None
            for ni, b in enumerate(nodes):
                at, et = cur[ni]
                base = self.cost(at, et)
                for k in range(self.nd):
                    lo, hi = b[k]
                    if hi - lo < 2:
                        continue
                    ca = self.along(b, self.lf, k).cumsum(axis=1)
                    ce = self.along(b, self.ef, k).cumsum(axis=1)
                    for t in range(1, hi - lo):
                        al = ca[:, t - 1]; el = ce[:, t - 1]
                        ar = at - al; er = et - el
                        if al.sum() < 1.0 or ar.sum() < 1.0:
                            continue
                        c = self.cost(al, el) + self.cost(ar, er)
                        gain = base - c
                        if best is None or gain > best[0]:
                            best = (gain, ni, k, lo + t)
            if best is None or best[0] <= 0:
                break
            gain, ni, k, t = best
            b = nodes[ni]
            lb = list(b); lb[k] = (b[k][0], t); lb = tuple(lb)
            rb = list(b); rb[k] = (t, b[k][1]); rb = tuple(rb)
            nodes[ni] = lb; nodes.append(rb)
            cur[ni] = (self.marg(lb, self.lf), self.marg(lb, self.ef))
            cur.append((self.marg(rb, self.lf), self.marg(rb, self.ef)))
            total -= gain
            curve.append((len(nodes), total))
            if verbose:
                print(f"    {len(nodes):2d} regimes  {total/1e6:9.4f} Mbit  f{k}<{t} gain {gain/1e6:.4f}", flush=True)
        return nodes, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats"); ap.add_argument("gid", type=int)
    ap.add_argument("--regs", type=int, default=16)
    ap.add_argument("--K", default="1000,4000,12000")
    ap.add_argument("--ref", required=True,
                    help="block_stats dump of the ENGINE 6 hand-tuned 8-regime baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    H, dims = load(a.stats, a.gid)
    R, rdims = load(a.ref, 0)
    print(f"grid {a.gid} [{GN[a.gid]}] dims {dims}")
    ntok = float(H[1].sum() + H[0].sum()) / 2
    for K in (float(x) for x in a.K.split(",")):
        cost = RegCost(K)
        # reference: ENGINE 6 hand-tuned 8 regimes
        ref = 0.0
        for lf, ef in ((0, 1), (1, 0)):
            for r in range(8):
                ref += cost(R[lf, :, r].astype(np.float64), R[ef, :, r].astype(np.float64))
        ref /= 2
        line = [f"K={K:.0f}  E6ref {ref/1e6:.3f} Mbit ({ref/ntok:.5f} b/sym)"]
        if a.gid != 0:
            for lf, ef in ((0, 1), (1, 0)):
                g = Grower(H, dims, K, lf, ef)
                nodes, curve = g.grow(a.regs, verbose=a.verbose)
                d = dict(curve)
                line.append(f"f{lf}{ef}: " + " ".join(f"{n}:{d[n]/1e6:.3f}" for n in sorted(d) if n in (2, 4, 8, 12, 16, 24, 32)))
                if lf == 0:
                    best = d
                else:
                    best = {n: (best[n] + d[n]) / 2 for n in set(best) & set(d)}
            line.append("MEAN: " + " ".join(f"{n}:{best[n]/1e6:.3f}({100*(ref-best[n])/ref:+.2f}%)"
                                           for n in sorted(best) if n in (4, 8, 12, 16, 24, 32)))
        print("  " + "\n  ".join(line))
    if a.out:
        K = float(a.K.split(",")[0])
        Hs = H.sum(axis=0, keepdims=True)
        g = Grower(Hs, dims, K, 0, 0)
        nodes, curve = g.grow(a.regs)
        lab = np.zeros(dims, dtype=np.int32)
        for i, b in enumerate(nodes):
            lab[sl_of(b)] = i
        # order regimes by run share (descending): screen-like first, photo-like last, so that
        # the trainer's neighbour-regime pooling (r-1, r, r+1) pools similar regimes
        Hs2 = Hs[0]
        share = []
        for i in range(len(nodes)):
            m = (lab == i)
            arr = Hs2[:, m].sum(axis=1)        # (grp, ctx, sym)
            tot = arr.sum()
            runs = arr.reshape(arr.shape[0], NCTX, 19, 12)[:, :, 1:, :].sum()
            share.append(float(runs) / max(float(tot), 1.0))
        order = np.argsort(-np.array(share))
        remap = np.zeros(len(nodes), dtype=np.int32)
        for new, old in enumerate(order):
            remap[old] = new
        lab = remap[lab]
        lut = lab.ravel().astype(np.uint8)
        lut.tofile(a.out.replace(".json", ".lut"))
        print("  regime order (run share):", " ".join(f"{share[o]:.3f}" for o in order))
        json.dump({"gid": a.gid, "dims": dims, "features": GN[a.gid], "regs": len(nodes),
                   "nodes": [list(map(list, b)) for b in nodes], "labels": lab.ravel().tolist(),
                   "K": K}, open(a.out, "w"))
        print(f"  wrote {a.out} ({len(nodes)} regimes)")


if __name__ == "__main__":
    main()
