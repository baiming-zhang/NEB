import numpy as np
def original_quadrature(level):
    a = (5 + 3 * np.sqrt(5)) / 20
    b = (5 - np.sqrt(5)) / 20
    bary = np.full((4, 4), b)
    np.fill_diagonal(bary, a)
    if level == 0:
        return (bary, np.full(4, 0.25))
    if level != 1:
        raise ValueError('Only original levels 0 and 1 are authorized.')
    points = []
    for index in range(4):
        vertices = np.eye(4)
        vertices[index] = 0.25
        points.extend(bary @ vertices)
    return (np.array(points), np.full(16, 1 / 16))

def forward_map(mapping, points):
    delta = points - mapping.center
    radius = np.linalg.norm(delta, axis=1)
    if np.any(radius <= 0) or not np.isfinite(radius).all():
        raise ValueError('Original integration point reaches an invalid radial center.')
    directions = delta / radius[:, None]
    mesh = mapping.mesh_ends(directions)
    true = mapping.true_ends(directions)
    physical, J, det, fraction = mapping.radial(directions, radius, mesh, true)
    back, Jback, _, _ = mapping.radial(directions, np.linalg.norm(physical - mapping.center, axis=1), true, mesh)
    receipt = dict(inverse_point_max=float(np.max(np.linalg.norm(back - points, axis=1))), inverse_J_max=float(np.max(np.abs(Jback @ J - np.eye(3)))), det_formula_max=float(np.max(np.abs(np.linalg.det(J) - det))), radial_fraction_min=float(fraction.min()), radial_fraction_max=float(fraction.max()), positive_det_min=float(det.min()), positive_det_max=float(det.max()))
    if receipt['inverse_point_max'] >= 1e-10 or receipt['inverse_J_max'] >= 1e-08 or (not np.isfinite(det).all()) or (receipt['positive_det_min'] <= 0) or (receipt['radial_fraction_min'] < -1e-09) or (receipt['radial_fraction_max'] > 1 + 1e-09):
        raise FloatingPointError(receipt)
    return (det, receipt)