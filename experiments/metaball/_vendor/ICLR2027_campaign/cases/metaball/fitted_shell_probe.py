@BilinearForm
def mass(u, v, w):
    return dot(u, v)

@BilinearForm
def stiffness(u, v, w):
    return dot(curl(u), curl(v))
