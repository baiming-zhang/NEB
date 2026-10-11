from cupyx.scipy.linalg import solve_triangular

def cholsolve(L, b):
    return solve_triangular(L.T, solve_triangular(L, b, lower=True), lower=False)
