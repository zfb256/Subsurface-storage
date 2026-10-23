"""MC-SSM-ABCD+: a mass-conserving ABCD+ water balance whose parameters come from a selective SSM."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# parallel: recharge split between the fast and slow stores; series: fast store drains partly to the slow store
STRUCTURE = os.environ.get("STRUCTURE", "parallel")

# Physical ranges for the per-step parameters.
RANGES = {
    "a": (0.80, 1.0),      # ABCD propensity for runoff before soil saturation
    "b": (50.0, 1500.0),   # ABCD upper limit of ET + soil storage (mm)
    "c": (0.0, 1.0),       # fraction of available water to recharge
    "f": (0.0, 1.0),       # parallel: fraction of recharge to the slow store; series: fraction of fast drainage to it
    "d1": (0.01, 1.0),     # fast store outflow fraction
    "d2": (0.0005, 0.1),   # slow store outflow fraction
    "ts": (-3.0, 1.0),     # all-snow temperature (C)
    "tr": (1.5, 5.0),      # all-rain / full-melt temperature (C)
}
NAMES = list(RANGES)
STATIC = {"ts", "tr"}       # snow thresholds stay time-invariant per basin


class SelectiveSSM(nn.Module):
    """Minimal Mamba-style (S6) block: input-dependent step size, B and C, diagonal A, sequential scan."""

    def __init__(self, d_in, d_model=32, d_state=16):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.dt = nn.Linear(d_model, d_model)
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_model, d_state)
        self.logA = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, d_state)).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):                      # x: (T, N, d_in)
        u = F.silu(self.inp(x))                # (T, N, H)
        dt = F.softplus(self.dt(u))            # (T, N, H)
        Bm, Cm = self.B(u), self.C(u)          # (T, N, K)
        A = -torch.exp(self.logA)              # (H, K)
        h = x.new_zeros(x.shape[1], u.shape[2], A.shape[1])
        ys = []
        for t in range(x.shape[0]):
            h = torch.exp(dt[t, :, :, None] * A) * h + dt[t, :, :, None] * Bm[t, :, None, :] * u[t, :, :, None]
            ys.append((h * Cm[t, :, None, :]).sum(-1) + self.D * u[t])
        return self.out(torch.stack(ys))       # (T, N, H)


class ParamNet(nn.Module):
    """Static part from catchment attributes, plus a bounded dynamic deviation from the SSM."""

    def __init__(self, n_attr, n_dyn, dynamic=True, d_model=32):
        super().__init__()
        self.dynamic = dynamic
        self.static = nn.Sequential(nn.Linear(n_attr, 64), nn.SiLU(), nn.Linear(64, len(NAMES)))
        if dynamic:
            self.ssm = SelectiveSSM(n_dyn + n_attr, d_model)
            self.head = nn.Linear(d_model, len(NAMES))
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.dyn_mask = torch.tensor([n not in STATIC for n in NAMES], dtype=torch.float32)

    def forward(self, attr, dyn):              # attr: (N, A), dyn: (T, N, Dd)
        z = self.static(attr)[None].expand(dyn.shape[0], -1, -1)
        if self.dynamic:
            a = attr[None].expand(dyn.shape[0], -1, -1)
            z = z + torch.tanh(self.head(self.ssm(torch.cat([dyn, a], -1)))) * 1.5 * self.dyn_mask
        s = torch.sigmoid(z)
        return {n: lo + (hi - lo) * s[..., i] for i, (n, (lo, hi)) in enumerate(RANGES.items())}


def water_balance(prm, p, pet, t, warm=120, init=None):
    """Run ABCD+ over (T, N) forcing. The first `warm` months only spin up the stores."""
    n = p.shape[1]
    if init is not None:
        snow, s, g1, g2 = init["snow"], init["soil"], init["g1"], init["g2"]
    else:
        snow = p.new_zeros(n)
        s = 0.5 * prm["b"][0]
        g1 = p.new_zeros(n)
        g2 = p.new_full((n,), 50.0)
    out = {k: [] for k in ("q", "soil", "g1", "g2", "snow", "et", "melt")}
    for i in range(p.shape[0]):
        a, b, c, f = prm["a"][i], prm["b"][i], prm["c"][i], prm["f"][i]
        frac = ((t[i] - prm["ts"][i]) / (prm["tr"][i] - prm["ts"][i])).clamp(0.0, 1.0)
        snow = snow + p[i] * (1.0 - frac)
        melt = snow * frac
        snow = snow - melt
        w = p[i] * frac + melt + s
        h = (w + b) / (2.0 * a)
        y = h - torch.sqrt((h * h - w * b / a).clamp_min(1e-9))
        y = torch.minimum(y, w)
        s = y * torch.exp(-pet[i] / b)
        avail = w - y
        rech = c * avail
        if STRUCTURE == "series":
            # all recharge enters the fast store; a fraction f of its drainage feeds the slow store
            g1 = g1 + rech
            drain = prm["d1"][i] * g1
            g1 = g1 - drain
            q1 = (1 - f) * drain
            g2 = g2 + f * drain
            q2 = prm["d2"][i] * g2
            g2 = g2 - q2
        else:
            g1 = g1 + (1 - f) * rech
            g2 = g2 + f * rech
            q1, q2 = prm["d1"][i] * g1, prm["d2"][i] * g2
            g1, g2 = g1 - q1, g2 - q2
        for k, v in (("q", (1 - c) * avail + q1 + q2), ("soil", s), ("g1", g1), ("g2", g2),
                     ("snow", snow), ("et", y - s), ("melt", melt)):
            out[k].append(v)
    return {k: torch.stack(v)[warm:] for k, v in out.items()}
