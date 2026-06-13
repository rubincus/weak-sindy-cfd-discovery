"""Spatial differentiators."""

from __future__ import annotations

import numpy as np

from weak_sindy_robust.grids import Grid1D, Grid2D


class FourierSpectral1D:
    """Fourier spectral derivatives on a periodic one-dimensional grid."""

    def __init__(self, grid: Grid1D):
        """Create derivative symbols for a grid."""
        self.grid = grid
        self.k = np.fft.fftfreq(grid.Nx, d=grid.dx) * 2.0 * np.pi
        self.ik = 1j * self.k
        self.minus_k2 = -(self.k**2)

    def dx(self, u: np.ndarray) -> np.ndarray:
        """Return the first derivative of one field row."""
        return np.real(np.fft.ifft(self.ik * np.fft.fft(u)))

    def dxx(self, u: np.ndarray) -> np.ndarray:
        """Return the second derivative of one field row."""
        return np.real(np.fft.ifft(self.minus_k2 * np.fft.fft(u)))

    def derivative(self, u: np.ndarray, order: int) -> np.ndarray:
        """Return the spatial derivative of arbitrary non-negative order."""
        if order == 0:
            return np.asarray(u, dtype=float)
        symbol = (1j * self.k) ** order
        return np.real(np.fft.ifft(symbol * np.fft.fft(u)))

    def dx_field(self, U: np.ndarray) -> np.ndarray:
        """Return first derivatives row by row."""
        return np.array([self.dx(row) for row in U])

    def dxx_field(self, U: np.ndarray) -> np.ndarray:
        """Return second derivatives row by row."""
        return np.array([self.dxx(row) for row in U])

    def derivative_field(self, U: np.ndarray, order: int) -> np.ndarray:
        """Return arbitrary derivatives row by row."""
        return np.array([self.derivative(row, order) for row in U])


class FourierSpectral2D:
    """Fourier spectral derivatives and Biot-Savart inversion on a periodic 2D grid."""

    def __init__(self, grid: Grid2D):
        """Create derivative symbols for a two-dimensional grid."""
        self.grid = grid
        kx = np.fft.fftfreq(grid.Nx, d=grid.dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(grid.Ny, d=grid.dy) * 2.0 * np.pi
        self.kx = kx[None, :]
        self.ky = ky[:, None]
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        self.k2 = self.kx**2 + self.ky**2
        self._inverse_k2 = np.zeros_like(self.k2, dtype=float)
        mask = self.k2 > 0.0
        self._inverse_k2[mask] = 1.0 / self.k2[mask]

    def _fft2(self, field: np.ndarray) -> np.ndarray:
        """Return the spatial FFT over the last two axes."""
        return np.fft.fft2(field, axes=(-2, -1))

    def _ifft2_real(self, field_hat: np.ndarray) -> np.ndarray:
        """Return the real inverse spatial FFT."""
        return np.real(np.fft.ifft2(field_hat, axes=(-2, -1)))

    def dx_field(self, U: np.ndarray) -> np.ndarray:
        """Return x derivatives for one field or a time stack."""
        return self._ifft2_real(self.ikx * self._fft2(U))

    def dy_field(self, U: np.ndarray) -> np.ndarray:
        """Return y derivatives for one field or a time stack."""
        return self._ifft2_real(self.iky * self._fft2(U))

    def laplacian_field(self, U: np.ndarray) -> np.ndarray:
        """Return Laplacians for one field or a time stack."""
        return self._ifft2_real(-self.k2 * self._fft2(U))

    def velocity_from_vorticity(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover incompressible velocity (u, v) from vorticity via streamfunction."""
        omega_hat = self._fft2(omega)
        psi_hat = omega_hat * self._inverse_k2
        u = self._ifft2_real(self.iky * psi_hat)
        v = self._ifft2_real(-self.ikx * psi_hat)
        return u, v

    def dealias(self, U: np.ndarray, fraction: float = 2.0 / 3.0) -> np.ndarray:
        """Apply a 2/3-style spectral low-pass filter to a field."""
        frac = float(np.clip(fraction, 0.0, 1.0))
        modes_x = np.abs(np.fft.fftfreq(self.grid.Nx))
        modes_y = np.abs(np.fft.fftfreq(self.grid.Ny))
        cutoff_x = frac * np.max(modes_x)
        cutoff_y = frac * np.max(modes_y)
        keep = (modes_y[:, None] <= cutoff_y) & (modes_x[None, :] <= cutoff_x)
        U_hat = self._fft2(U)
        return self._ifft2_real(U_hat * keep)
