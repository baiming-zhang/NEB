import cupy as cp
import mfs
EXT_POWER=2
def smoothstep(t):
    t = cp.clip(t, 0, 1)
    return (t ** 6 * (462 + t * (-1980 + t * (3465 + t * (-3080 + t * (1386 - 252 * t))))), 2772 * t ** 5 * (1 - t) ** 5)

def extension(x, geo, c):
    out, inner, go, gi = mfs.levels(x)

    def factor(q, g, a):
        t = q / a
        negative = cp.minimum(t, 0)
        den = 1 + t + negative ** EXT_POWER
        v = t / den
        dv = (den - t * (1 + EXT_POWER * negative ** (EXT_POWER - 1))) / den ** 2 / a
        return (v, dv[:, None] * g)
    fo, dfo = factor(out, go, c.outer_envelope_transition)
    fi, dfi = factor(inner, gi, c.obstacle_envelope_transition)
    s, ds = smoothstep((-inner - 0.04) / 0.04)
    w = 1 - s
    dw = -ds[:, None] * gi * (-1 / 0.04)
    dfi = dfi * w[:, None] + fi[:, None] * dw
    fi *= w
    e = fo * fi
    de = dfo * fi[:, None] + dfi * fo[:, None]
    for j in range(3):
        distance = cp.maximum(-0.03 - x[:, j], x[:, j] - 1.03)
        sign = cp.where(x[:, j] < 0.5, -1.0, 1.0)
        s, ds = smoothstep(distance / 0.03)
        w = 1 - s
        dw = -ds * sign / 0.03
        de *= w[:, None]
        de[:, j] += e * dw
        e *= w
    return Jet1(e[:, None], de[:, None, :])