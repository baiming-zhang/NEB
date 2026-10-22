










def _terms(cp, a, d, h, lam, eta):
    a = cp.asarray(a, dtype="f8")
    d = cp.asarray(d, dtype="f8")
    h = cp.asarray(h, dtype="f8")
    lam = cp.asarray(lam, dtype="f8")
    eta = cp.asarray(eta, dtype="f8")
    n, k = d.shape
    assert n > 0 and k > 0
    assert a.shape == (3 * n, k) and h.shape == (3 * n, 1)
    assert lam.shape == (n, k) and eta.shape == (k,)
    assert all(bool(cp.all(cp.isfinite(v))) for v in (a, d, h, lam, eta))

    mass = cp.sum(a * a, axis=0) / n
    sh2 = cp.sum(h * h) / n
    assert bool(cp.all(mass > 0)) and float(sh2) > 0
    s = cp.sqrt(mass)
    sh = cp.sqrt(sh2)
    r = d / s[None, :]
    moment = (h.T @ a).reshape(-1) / (n * s * sh)

    m = a.T @ a / n
    b = d.T @ d / n

    m = (m + m.T) * 0.5
    b = (b + b.T) * 0.5
    eigenvalues = cp.linalg.eigvalsh(m)
    assert bool(cp.all(cp.isfinite(eigenvalues)))
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    assert largest > 0 and smallest > largest * 1e-12, (
        "Potential Gram is not full SPD at the declared relative 1e-12 gate; "
        "no ridge, truncation, or clipping is allowed"
    )
    inverse = cp.linalg.solve(m, cp.eye(k, dtype="f8"))
    inverse = (inverse + inverse.T) * 0.5
    span_div_energy = cp.trace(inverse @ b)
    assert bool(cp.isfinite(span_div_energy)) and float(span_div_energy) >= 0
    return a, d, h, lam, eta, n, k, mass, s, sh, r, moment, b, inverse, {
        "cond": largest / smallest,
        "minimum_potential_Gram_eigenvalue": smallest,
        "maximum_potential_Gram_eigenvalue": largest,
        "relative_SPD_gate": 1e-12,
        "rank": k,
        "modes_removed": 0,
        "ridge": 0.0,
        "span_div_energy": float(span_div_energy),
        "span_div_energy_per_mode": float(span_div_energy) / k,
        "precision": "FP64",
        "metric": "tr(M_A^-1 M_div); full joint Gram adjoint",
        "multiplier_coordinates": "Original normalized raw mode columns",
    }


def span_constraint_functional(cp, a, d, h, lam, eta, rho, scale):





    assert float(rho) >= 0 and float(scale) >= 0
    terms = _terms(cp, a, d, h, lam, eta)
    _, _, _, lam, eta, _, _, _, _, _, r, moment, b, inverse, _ = terms
    return (
        cp.sum(cp.mean(lam * r, axis=0))
        + scale * cp.sum(eta * moment + 0.5 * rho * moment * moment)
        + 0.5 * rho * cp.trace(inverse @ b)
    )


def span_augmented(cp, a, d, h, lam, eta, rho, scale):






    assert float(rho) >= 0 and float(scale) >= 0
    (
        a, d, h, lam, eta, n, k, mass, s, sh, r, moment, b, inverse, info
    ) = _terms(cp, a, d, h, lam, eta)

    linear_div = cp.mean(lam * r, axis=0)
    weak_term = scale * (eta * moment + 0.5 * rho * moment * moment)
    span_energy = cp.trace(inverse @ b)
    cost = linear_div + weak_term + 0.5 * rho * span_energy / k


    gd = lam / (n * s[None, :])
    ga = -a * (linear_div / (n * mass))[None, :]


    gd += rho * (d @ inverse) / n
    gram_cotangent = inverse @ b @ inverse
    gram_cotangent = (gram_cotangent + gram_cotangent.T) * 0.5
    ga -= rho * (a @ gram_cotangent) / n


    weak_multiplier = scale * (eta + rho * moment)
    ga += weak_multiplier[None, :] * (
        h / (n * s[None, :] * sh)
        - a * (moment / (n * mass))[None, :]
    )
    assert bool(cp.all(cp.isfinite(cost)))
    assert bool(cp.all(cp.isfinite(ga))) and bool(cp.all(cp.isfinite(gd)))
    info.update(
        divergence_quadratic_cost=float(0.5 * rho * span_energy),
        normalized_raw_multiplier_cost=float(cp.sum(linear_div)),
        weak_wall_cost=float(cp.sum(weak_term)),
        total_constraint_cost=float(cp.sum(cost)),
    )
    return cost, ga, gd, r, moment, info
