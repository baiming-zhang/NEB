"""Original consistent-mass P and harmonic-completed PH finite-element controls."""
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from .baseline_dense import TopologyNS
def scalar(value) -> float:
    return float(value.item())

def solve_positive_modes(model: TopologyNS, count: int):
    chol_m = cp.linalg.cholesky(model.Mf)
    left = solve_triangular(chol_m, model.Sf, lower=True)
    standard = solve_triangular(chol_m, left.T, lower=True).T
    standard = 0.5 * (standard + standard.T)
    values, vectors = cp.linalg.eigh(standard)
    values = values[:count]
    vectors = vectors[:, :count]
    free = solve_triangular(chol_m.T, vectors, lower=False)
    potentials = cp.zeros((model.n, count), dtype=cp.float64)
    potentials[model.F] = free
    potentials /= cp.sqrt(values)[None, :]
    columns = cp.arange(count)
    anchors = cp.argmax(cp.abs(potentials), axis=0)
    potentials *= cp.where(potentials[anchors, columns] < 0.0, -1.0, 1.0)[None, :]
    residuals = []
    for index in range(count):
        vector = potentials[model.F, index]
        residual = model.Sf @ vector - values[index] * (model.Mf @ vector)
        scale = cp.linalg.norm(model.Sf @ vector)
        residuals.append(scalar(cp.linalg.norm(residual) / cp.maximum(scale, 1e-30)))
    return (values, potentials, residuals)

def energy_orthonormal_harmonics(model: TopologyNS):
    chol = cp.linalg.cholesky(model.harmonic_gram)
    transform = solve_triangular(chol.T, cp.eye(model.harmonic_dimension, dtype=cp.float64), lower=False)
    basis = model.H @ transform
    return basis

def basis_diagnostics(model: TopologyNS, basis, rates, eigen_residuals=None) -> dict:
    gram = basis.T @ model.S @ basis
    boundary_trace = scalar(cp.max(cp.abs(basis[model.D]))) if bool(cp.all(rates > 0).item()) else None
    return {'dimension': int(basis.shape[1]), 'energy_gram_max_abs_error': scalar(cp.max(cp.abs(gram - cp.eye(len(rates))))), 'positive_boundary_trace_max': boundary_trace, 'eigen_residual_max': max(eigen_residuals) if eigen_residuals else None}

def project_field(model: TopologyNS, basis, rates, psi, omega, gamma) -> dict:
    coefficients = basis.T @ model.S @ psi
    reconstruction = basis @ coefficients
    omega_reconstruction = basis[:, rates > 0] @ (coefficients[rates > 0] * rates[rates > 0])
    error = reconstruction - psi
    omega_error = omega_reconstruction - omega
    norm2 = psi @ model.S @ psi
    omega_norm2 = omega @ model.M @ omega
    reconstructed_gamma = model.gamma_from_state(reconstruction, omega_reconstruction)
    trace_gt = cp.stack([cp.mean(psi[nodes]) for nodes in model.boundary_nodes[1:]])
    trace_rec = cp.stack([cp.mean(reconstruction[nodes]) for nodes in model.boundary_nodes[1:]])
    return {'relative_velocity_L2': scalar(cp.sqrt(error @ model.S @ error / cp.maximum(norm2, 1e-30))), 'relative_vorticity_L2': scalar(cp.sqrt(omega_error @ model.M @ omega_error / cp.maximum(omega_norm2, 1e-30))), 'kinetic_energy_capture': scalar(reconstruction @ model.S @ reconstruction / cp.maximum(norm2, 1e-30)), 'inner_trace_L2_error': scalar(cp.linalg.norm(trace_rec - trace_gt)), 'weak_gamma_L2_error': scalar(cp.linalg.norm(reconstructed_gamma - gamma)), 'coefficients': cp.asnumpy(coefficients), 'psi': cp.asnumpy(reconstruction)}

def generate(mesh,total_modes,include_harmonics=True,viscosity=1/750):
    """Original dense P1 controls: P_K or P_(K-h)+H_h, with full consistent mass."""
    model=TopologyNS(mesh,viscosity)
    h=model.harmonic_dimension if include_harmonics else 0
    if not h<total_modes<len(model.F):raise ValueError('Mode count outside full free-node space')
    rates,positive,residuals=solve_positive_modes(model,total_modes-h)
    harmonics=energy_orthonormal_harmonics(model) if h else cp.zeros((model.n,0),dtype=cp.float64)
    basis=cp.column_stack((positive,harmonics));all_rates=cp.concatenate((rates,cp.zeros(h)))
    return model,basis,all_rates,basis_diagnostics(model,basis,all_rates,residuals)
