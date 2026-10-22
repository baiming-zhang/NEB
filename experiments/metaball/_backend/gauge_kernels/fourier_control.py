import numpy as np,math,itertools
class ReusePoisson(Poisson):

    def prepare(self, source):
        self.fh = cp.fft.fftshift(cp.fft.fftn(cp.asarray(source).reshape((self.n,) * 3))) / self.n ** 3
        self.mean = self.fh[self.mid].real.copy()
        self.coef = -self.fh / self.kk
        self.coef[self.mid] = 0

    def evaluate(self, x, kind):
        angles = (x - LO) * (2 * np.pi / LENGTH)
        r = x - 0.5
        if kind == 'value':
            return native_nufft.evaluate(self.coef, angles).real + self.mean * cp.sum(r * r, axis=1) / 6
        if kind == 'lap':
            return native_nufft.evaluate(self.fh, angles).real
        return cp.stack([native_nufft.evaluate(1j * k * self.coef, angles).real + self.mean * r[:, j] / 3 for j, k in enumerate(self.k)], axis=1)

    def spectrum_transpose(self, x, value=None, gradient=None):
        angles = (x - LO) * (2 * np.pi / LENGTH)
        r = x - 0.5
        a = cp.zeros((self.n,) * 3, 'c16')
        mean = cp.zeros(())
        if value is not None:
            a += native_nufft.adjoint(value, angles, self.n)
            mean += cp.sum(value * cp.sum(r * r, axis=1) / 6)
        if gradient is not None:
            for j, k in enumerate(self.k):
                a -= 1j * k * native_nufft.adjoint(gradient[:, j], angles, self.n)
            mean += cp.sum(gradient * r / 3)
        a /= -self.kk
        a[self.mid] = mean
        return a

    def transpose_combined(self, x, gradient, xb, value):
        a = self.spectrum_transpose(x, gradient=gradient) + self.spectrum_transpose(xb, value=value)
        return cp.fft.ifftn(cp.fft.ifftshift(a)).real.ravel().copy()
