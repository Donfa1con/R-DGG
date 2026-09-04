"""The directed Granger gain of R-DGG: design assembly, the GT reference operator, and the residual gain.

Pure tensor math on the pooled design. No corpus paths are read here.
"""
import numpy as np
import torch

from _config import DEV, DT, RIDGE, SCHUR_RIDGE


def standardize_cols(X):
    mu = X.mean(0)
    sd = torch.maximum(X.std(0, correction=0), 1e-6 * (X.abs().mean(0) + 1e-12))
    return (X - mu) / sd


def standardize_lagged(X, keep):
    """Standardise a lag-major design on the entries that carry data, then blank the rest.

    `keep` is (n_rows, n_lags, 1), zero where a lag reaches back past the start of that row's listening segment. Those
    entries are not merely unavailable, they are WRONG: the gather is modulo the source length, so a lag reaching past the
    segment reads unrelated material from elsewhere in the recording.

    Two naive treatments are both wrong. Masking first puts structural zeros inside the column's own mean and standard
    deviation, and since the deep lags are unavailable far more often than the shallow ones -- 40% of rows at 4 s
    against 0% below 0.44 s -- that deflates their scale by 23% at the deepest lag and so inflates those columns by
    about 1.29x after division, which relative ridge shrinkage then under-penalises. Standardising over all rows first
    puts the contaminated values in instead. Estimating on the available entries alone and blanking afterwards is the
    treatment that depends on neither how often a lag was reachable nor on what it read when it was not.
    """
    n_lags = keep.shape[1]
    Xr = X.reshape(X.shape[0], n_lags, -1)
    n = keep.sum(0).clamp_min(1.0)
    mu = (Xr * keep).sum(0) / n
    var = (((Xr - mu) ** 2) * keep).sum(0) / n
    sd = torch.maximum(var.sqrt(), 1e-6 * ((Xr.abs() * keep).sum(0) / n + 1e-12))
    out = Xr - mu                 # fresh buffer (Xr is a view of X, never mutated); reuse it in place below
    out /= sd; out *= keep        # == ((Xr - mu) / sd) * keep, but one alloc not three (peak ~5 GB not ~15)
    return out.reshape(X.shape[0], -1), mu, sd


def apply_lagged_standardisation(X, mu, sd, keep):
    """Re-apply a standardisation estimated elsewhere: the operator must see the design it was fitted on."""
    n_lags = keep.shape[1]
    Xr = X.reshape(X.shape[0], n_lags, -1)
    out = Xr - mu                 # fresh buffer (Xr is a view of X, never mutated); reuse it in place below
    out /= sd; out *= keep        # == ((Xr - mu) / sd) * keep, but one alloc not three (peak ~5 GB not ~15)
    return out.reshape(X.shape[0], -1)


def own_audio_block(ctx, order, tids):
    """The listener's OWN speech past, lagged: the half of the control that absorbs turn-taking.

    One implementation. The reference-operator fit and the model scoring both build this block, and it is what stops
    the metric rewarding a pure turn-taking readout, so it must be ONE version -- two could drift apart silently.

    Returned UNMASKED. The mask belongs after standardisation, which is estimated on the reachable entries alone -- see
    standardize_lagged.
    """
    k = ctx.corpus.k_audio
    lag = ctx.corpus.LAG
    blocks = []
    for ls, tid in zip(order, tids):
        lis_audio = ctx.driver[ls]
        idx = (tid[:, None] - lag[None, :]) % lis_audio.shape[0]
        blocks.append(lis_audio[idx][:, :, :k].reshape(idx.shape[0], -1))
    return torch.cat(blocks, 0)


def own_block_standardised(ctx, order, tids, motion, keep):
    """The control: the listener's own motion and own speech past, each standardised on its reachable entries, then masked.

    The two halves are standardised SEPARATELY because they have different widths per lag, so the concatenation is not
    lag-major and cannot be reshaped as one.
    """
    z_motion, _, _ = standardize_lagged(motion, keep)
    z_audio, _, _ = standardize_lagged(own_audio_block(ctx, order, tids), keep)
    return torch.cat([z_motion, z_audio], 1)


def aud_given_mot(both, motion_only):
    """The single reported channel: the speaker AUDIO gain beyond the speaker's own MOTION, as both - motion_only.

    Type-agnostic on purpose -- called on floats inside the bootstrap, on the numpy null stack, and on the per-video
    vectors in the printer -- so there is one implementation of the difference rather than three that agree by
    inspection.
    """
    return both - motion_only


def row_lag_frames(tids, lag):
    """For every design row, the source frame each lag reaches: (n_rows, n_lags)."""
    return torch.cat([t[:, None] - lag[None, :] for t in tids], 0)


def lag_is_in_segment(depths, lag):
    """(n_rows, n_lags, 1): zero where a lag would reach back past the start of that row's listening segment."""
    return (lag[None, :] <= torch.cat(depths)[:, None]).to(DT)[:, :, None]


def gather_driver(ctx, order, spk_map, off_map, frames, rows_per_listener):
    """The speaker's driver features at every lag, for every design row: (n_rows, n_lags * driver_width).

    ONE implementation, called both where the reference operator is FITTED and where it is APPLIED. Fit and apply MUST
    gather identically -- same modulo, same mask, same lag-major layout -- or the operator would be applied to a design
    it was not fitted on with nothing raising, so a single shared implementation is the safe form. The global gather is
    the shared one because applying runs hundreds of times per model against four fits, so the fast path is worth having
    in both places.

    `frames` and `rows_per_listener` are arguments rather than derived here: the applying side caches them across calls.
    """
    speakers = [spk_map[ls] for ls in order]
    row_n = torch.repeat_interleave(
        torch.tensor([ctx.driver[s].shape[0] for s in speakers], device=DEV), rows_per_listener
    )
    row_start = torch.repeat_interleave(
        torch.tensor([ctx.speaker_row0[s] for s in speakers], device=DEV), rows_per_listener
    )
    row_off = torch.repeat_interleave(
        torch.tensor([off_map.get(ls, 0) if off_map else 0 for ls in order], device=DEV), rows_per_listener
    )
    gidx = row_start[:, None] + ((frames - row_off[:, None]) % row_n[:, None])
    return ctx.driver_all[gidx].reshape(gidx.shape[0], -1)


GRAM_CHUNK = 16384  # fp64-Gram streaming block; result is numerically identical to any value — larger
                    # means fewer, bigger GEMMs (less launch overhead). Free to raise: 16384 adds ~0.6 GiB
                    # fp64 transient, well within headroom.


def gram_fp64(Xz, rhs=None):
    d = Xz.shape[1]
    G = torch.zeros(d, d, device=DEV, dtype=torch.float64)
    R = None if rhs is None else torch.zeros(d, rhs.shape[1], device=DEV, dtype=torch.float64)
    for i in range(0, Xz.shape[0], GRAM_CHUNK):
        blk = Xz[i : i + GRAM_CHUNK].double()
        G += blk.T @ blk
        if R is not None:
            R += blk.T @ rhs[i : i + GRAM_CHUNK].double()
    return G, R


def gram_fp64_cat(Xo, Xs, rhs):
    """Gram and rhs of the concatenation [Xo | Xs] WITHOUT materialising the concat (which is a full fp32
    copy of both blocks -- ~13 GB on the large corpus, and the sole thing that OOMs the B-fit at 44 GB).
    Numerically IDENTICAL to gram_fp64(torch.cat([Xo, Xs], 1), rhs): the same per-chunk fp64 products
    (blk.double() @ blk) accumulated in the same order; the concat's blk.T@blk is exactly these sub-blocks."""
    do, ds = Xo.shape[1], Xs.shape[1]; d = do + ds
    G = torch.zeros(d, d, device=DEV, dtype=torch.float64)
    R = torch.zeros(d, rhs.shape[1], device=DEV, dtype=torch.float64)
    for i in range(0, Xo.shape[0], GRAM_CHUNK):
        o = Xo[i : i + GRAM_CHUNK].double(); s = Xs[i : i + GRAM_CHUNK].double(); y = rhs[i : i + GRAM_CHUNK].double()
        G[:do, :do] += o.T @ o
        G[:do, do:] += o.T @ s
        G[do:, do:] += s.T @ s
        R[:do] += o.T @ y
        R[do:] += s.T @ y
    G[do:, :do] = G[:do, do:].T                                   # symmetric off-diagonal (no re-accumulation)
    return G, R


def cross_gram_fp64(Xo, Xs, Y):
    """Xo.T@Xs, Xs.T@Xs and Xs.T@Y with an fp64 accumulator but fp32 products.

    Different arithmetic from gram_fp64 above, and deliberately: this runs ~350 times per model, and the wide product
    is (4750 x 252), which in full fp64 would cost ~0.5 s a call at this GPU's 1/64 double rate. Promoting only the
    ACCUMULATOR removes the cross-chunk rounding -- the dominant term at 3e5 rows -- for the price of a few small adds.
    gram_fp64 stays fully fp64 because it runs once per model and its matrix is the one that has to factorise.
    """
    A_os = torch.zeros(Xo.shape[1], Xs.shape[1], device=DEV, dtype=torch.float64)
    G_ss = torch.zeros(Xs.shape[1], Xs.shape[1], device=DEV, dtype=torch.float64)
    rhs_s = torch.zeros(Xs.shape[1], Y.shape[1], device=DEV, dtype=torch.float64)
    for i in range(0, Xo.shape[0], GRAM_CHUNK):
        o, s, y = Xo[i : i + GRAM_CHUNK], Xs[i : i + GRAM_CHUNK], Y[i : i + GRAM_CHUNK]
        A_os += (o.T @ s).double()
        G_ss += (s.T @ s).double()
        rhs_s += (s.T @ y).double()
    return A_os, G_ss, rhs_s


def _ridge_solve(Xz, Yc):
    dim = Xz.shape[1]
    G, rhs = gram_fp64(Xz, Yc)
    lam = RIDGE * torch.diagonal(G).mean()
    A = G + lam * torch.eye(dim, device=DEV, dtype=torch.float64)
    return torch.linalg.solve(A, rhs).to(DT)


def pool_design(per, consume=False):
    order = list(per.keys())
    tids = [torch.as_tensor(per[ls][2], device=DEV, dtype=torch.long) for ls in order]
    depths = [torch.as_tensor(per[ls][3], device=DEV, dtype=torch.long) for ls in order]
    if not consume:
        XO = torch.cat([torch.as_tensor(per[ls][0], device=DEV, dtype=DT) for ls in order], 0)
        Y = torch.cat([torch.as_tensor(per[ls][1], device=DEV, dtype=DT) for ls in order], 0)
        return order, XO, Y, tids, depths
    n_rows = sum(int(per[ls][0].shape[0]) for ls in order)
    XO = torch.empty((n_rows, int(per[order[0]][0].shape[1])), device=DEV, dtype=DT)
    Y = torch.empty((n_rows, int(per[order[0]][1].shape[1])), device=DEV, dtype=DT)
    row0 = 0
    for ls in order:
        design, target = per[ls][0], per[ls][1]
        n_stem_rows = int(design.shape[0])
        XO[row0 : row0 + n_stem_rows].copy_(torch.as_tensor(design, device=DEV, dtype=DT))
        Y[row0 : row0 + n_stem_rows].copy_(torch.as_tensor(target, device=DEV, dtype=DT))
        per[ls] = None
        row0 += n_stem_rows
    return order, XO, Y, tids, depths


def fit_bgt_fold(gt_per, holdout, ctx):
    sub = {ls: val for ls, val in gt_per.items() if ctx.corpus.FOLD[ls] != holdout}
    if len(sub) < 5:
        return None
    order, XO, Y, tids, depths = pool_design(sub)
    lag = ctx.corpus.LAG
    keep = lag_is_in_segment(depths, lag)
    rows_per_listener = torch.tensor([int(t.shape[0]) for t in tids], device=DEV)
    XA = gather_driver(ctx, order, ctx.corpus.pairs, None, row_lag_frames(tids, lag), rows_per_listener)
    Yc = Y - Y.mean(0)
    if ctx.target_scale_inv is not None:                 # balance target dims (EMOCA vs LP) before fitting B
        Yc = Yc * ctx.target_scale_inv
    Xz_o = own_block_standardised(ctx, order, tids, XO, keep)
    d_o = Xz_o.shape[1]
    XAz, mu_a, sd_a = standardize_lagged(XA, keep)

    def driver_coeffs_of(driver):                              # = _ridge_solve(cat([Xz_o, driver]))[d_o:]
        G, rhs = gram_fp64_cat(Xz_o, driver, Yc)               # but block Gram -> no ~13 GB fp32 concat copy
        lam = RIDGE * torch.diagonal(G).mean()
        A = G + lam * torch.eye(G.shape[0], device=DEV, dtype=torch.float64)
        return torch.linalg.solve(A, rhs).to(DT)[d_o:]

    def fit_driver_block(cols):
        return driver_coeffs_of(XAz[:, cols])

    Bd = {chan: fit_driver_block(cols) for chan, cols in ctx.channel_cols.items()}
    return mu_a, sd_a, Bd


class DriverPredictor:
    """s_hat: what GT's reaction operator predicts for these rows, given a speaker pairing and optional time shifts.

    Holds the row bookkeeping that every prediction reuses -- which fold each row belongs to, which lag frames it reads,
    where each speaker starts in the stacked driver -- so a prediction is one gather and two matmuls per fold, one
    per channel block.
    """

    def __init__(self, ops, ctx, order, tids, depths):
        self.ops, self.ctx, self.order = ops, ctx, order
        n_fold = ctx.corpus.NFOLD
        lag = ctx.corpus.LAG
        self.region_width = ops[0][2]["audio"].shape[1]
        self.driver_mu = torch.stack([ops[f][0] for f in range(n_fold)])
        self.driver_sd = torch.stack([ops[f][1] for f in range(n_fold)])
        self.row_fold = torch.cat(
            [
                torch.full((int(t.shape[0]),), ctx.corpus.FOLD[ls], device=DEV, dtype=torch.long)
                for ls, t in zip(order, tids)
            ]
        )
        self.row_lag_frame = row_lag_frames(tids, lag)
        self.rows_per_listener = torch.tensor([int(t.shape[0]) for t in tids], device=DEV)
        self.rows_of_fold = [(self.row_fold == f).nonzero(as_tuple=True)[0] for f in range(n_fold)]
        self.audio_cols = ctx.channel_cols["audio"].nonzero(as_tuple=True)[0]
        self.motion_cols = ctx.channel_cols["motion"].nonzero(as_tuple=True)[0]
        self.lag_is_in_segment = lag_is_in_segment(depths, lag)

    def predict(self, spk_map, off_map=None):
        ctx, order = self.ctx, self.order
        raw = gather_driver(ctx, order, spk_map, off_map, self.row_lag_frame, self.rows_per_listener)
        std = apply_lagged_standardisation(
            raw, self.driver_mu[self.row_fold], self.driver_sd[self.row_fold], self.lag_is_in_segment
        )
        out = {
            "audio": torch.empty((std.shape[0], self.region_width), device=DEV, dtype=DT),
            "motion": torch.empty((std.shape[0], self.region_width), device=DEV, dtype=DT),
        }
        for fold in range(ctx.corpus.NFOLD):
            fi = self.rows_of_fold[fold]
            Bd = self.ops[fold][2]
            sm = std.index_select(0, fi)
            out["audio"].index_copy_(0, fi, sm.index_select(1, self.audio_cols) @ Bd["audio"])
            out["motion"].index_copy_(0, fi, sm.index_select(1, self.motion_cols) @ Bd["motion"])
        return out


class ControlFit:
    """The model's own-past baseline, fitted once per model, plus the statistic measured against it.

    `residual_with` adds predicted driver blocks on top of the fixed own-block via a Schur solve, so the own
    factorisation is reused instead of refactorising the whole system per channel and per null.
    """

    def __init__(self, Xz_o, Y, order, rows_per_listener, target_scale_inv=None):
        self.n_rows = Xz_o.shape[0]
        self.Yc = Y - Y.mean(0)
        if target_scale_inv is not None:                 # same target balancing as the B fit (EMOCA vs LP)
            self.Yc = self.Yc * target_scale_inv
        self.Xz_o = Xz_o
        self.d_o = self.Xz_o.shape[1]
        G_oo, rhs_o = gram_fp64(self.Xz_o, self.Yc)
        lam_o = RIDGE * torch.diagonal(G_oo).mean()
        A_oo = G_oo + lam_o * torch.eye(self.d_o, device=DEV, dtype=torch.float64)
        self.L_oo, info = torch.linalg.cholesky_ex(A_oo)
        assert not info.any(), (
            f"own-block Gram is not positive definite (cholesky info {int(info.max())}): the design is rank deficient "
            f"or a source produced constant dims. RIDGE is {RIDGE} and is NOT the thing to change."
        )
        self.beta_o0 = torch.cholesky_solve(rhs_o, self.L_oo)
        R_own = self.Yc - self.Xz_o @ self.beta_o0.to(DT)
        floor = torch.tensor(1e-30, device=DEV, dtype=DT)
        self.own_r2 = float(1.0 - (R_own**2).sum() / (self.Yc**2).sum().clamp_min(floor))
        self.energy_own = (R_own * R_own).sum(1)               # energy = the (target-norm-balanced) squared residual norm
        self.log_floor = torch.tensor(1e-12, device=DEV, dtype=DT)
        self.energy_own_total = self.energy_own.sum().clamp_min(self.log_floor)
        self.row_listener = torch.repeat_interleave(torch.arange(len(order), device=DEV), rows_per_listener)
        self.energy_own_per_listener = torch.zeros(len(order), device=DEV, dtype=DT).index_add_(
            0, self.row_listener, self.energy_own
        )

    def row_energy(self, resid):
        return (resid * resid).sum(1)

    def residual_with(self, blocks):
        Xz_s = torch.cat([standardize_cols(b) for b in blocks], 1)
        A_os, G_ss, rhs_s = cross_gram_fp64(self.Xz_o, Xz_s, self.Yc)
        lam_s = SCHUR_RIDGE * torch.diagonal(G_ss).mean()
        A_ss = G_ss + lam_s * torch.eye(Xz_s.shape[1], device=DEV, dtype=torch.float64)
        M = torch.cholesky_solve(A_os, self.L_oo)
        schur = A_ss - A_os.T @ M
        L_s, info = torch.linalg.cholesky_ex(schur)
        assert not info.any(), (
            f"schur complement is not positive definite (cholesky info {int(info.max())}): the driver block is "
            f"collinear with the own block beyond what the ridge spans. Arithmetic or a degenerate design, NOT a "
            f"reason to raise RIDGE."
        )
        beta_s = torch.cholesky_solve(rhs_s - A_os.T @ self.beta_o0, L_s)
        beta_o = self.beta_o0 - M @ beta_s
        return self.Yc - self.Xz_o @ beta_o.to(DT) - Xz_s @ beta_s.to(DT)

    def gain(self, resid):
        return torch.log(self.energy_own_total / self.row_energy(resid).sum().clamp_min(self.log_floor))

    def gain_per_listener(self, resid):
        per = torch.zeros(self.energy_own_per_listener.shape[0], device=DEV, dtype=DT).index_add_(
            0, self.row_listener, self.row_energy(resid)
        )
        return torch.log(self.energy_own_per_listener.clamp_min(1e-12) / per.clamp_min(1e-12))


def capture_energies(per, ops, ctx, timeshifts=None):
    """Per-listener residual energies under the fixed GT operator -- the raw material BOTH tables share.

    aud|mot net = both - motion_only = log(e_mot/e_both), so the model's own-past energy cancels and the four
    per-listener sums {both_r, mot_r, both_n[K], mot_n[K]} determine every aud|mot figure (point, floor, per-video
    net, cluster bootstrap).
    Returns ({stem: {...}}, own_r2); own_r2 is a ControlFit diagnostic, not aud|mot.
    """
    order, XO, Y, tids, depths = pool_design(per, consume=True)
    pred = DriverPredictor(ops, ctx, order, tids, depths)
    Xz_o = own_block_standardised(ctx, order, tids, XO, pred.lag_is_in_segment)
    del XO
    fit = ControlFit(Xz_o, Y, order, pred.rows_per_listener, target_scale_inv=ctx.target_scale_inv)
    n_ls = len(order)
    rl = fit.row_listener

    def per_listener(row_energy):                          # (n_rows,) -> (n_listeners,) numpy, summed by listener
        return torch.zeros(n_ls, device=DEV, dtype=DT).index_add_(0, rl, row_energy).detach().cpu().numpy()

    def resid_energy(off):
        shat = pred.predict(ctx.corpus.pairs, off)
        e_both = fit.row_energy(fit.residual_with([shat["audio"], shat["motion"]]))
        e_mot = fit.row_energy(fit.residual_with([shat["motion"]]))
        return per_listener(e_both), per_listener(e_mot)

    both_r, mot_r = resid_energy(None)
    ts = ctx.corpus.timeshifts if timeshifts is None else timeshifts   # paired 8x8 passes a reduced set (fast)
    K = len(ts)
    both_n = np.empty((K, n_ls)); mot_n = np.empty((K, n_ls))
    for k in range(K):                                     # one null resident at a time (K predicts, reduced on the fly)
        both_n[k], mot_n[k] = resid_energy(ts[k])
    nrows = np.bincount(rl.detach().cpu().numpy(), minlength=n_ls)
    E = {stem: {"both_r": float(both_r[i]), "mot_r": float(mot_r[i]),
                "both_n": both_n[:, i].astype(np.float64), "mot_n": mot_n[:, i].astype(np.float64),
                "nrows": int(nrows[i])}
         for i, stem in enumerate(order)}
    return E, fit.own_r2
