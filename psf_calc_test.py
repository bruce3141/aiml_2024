# psf_calc_test_2.py
"""
Created on Sep 20 1:05 PM 2023
@author: brucedean
"""

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

# Set the linewidth to avoid wrapping the display of large matrices
np.set_printoptions(linewidth=300)


def format_number(number, decimal_places=2):
    # Makes the display of numbers in the scientific format easier to read (fewer digits)
    # Convert the number to a string
    str_num = str(number)
    # Check if the number is in scientific notation
    if 'e' in str_num or 'E' in str_num:
        return f"{number:.{decimal_places}e}"
    else:
        return f"{number:.{decimal_places}f}"


def dft2d(x: np.ndarray, N: int) -> np.ndarray:
    """Compute the 2D DFT manually with output size N x N. """
    n, m = x.shape
    k = np.arange(N)
    l = np.arange(N)
    e_n = np.exp(-2j * np.pi * np.outer(k, np.arange(n)) / N)
    e_m = np.exp(-2j * np.pi * np.outer(l, np.arange(m)) / N)
    return np.dot(np.dot(e_n, x), np.conjugate(e_m).T)


class ZernikePolynomials:
    """
    Uses ordering for the Malacara basis set.
    Order-4:
    ------------------------------
    Z_1 = 1
    ------------------------------
    Z_2 = √2 × (r) × sin(θ)
    Z_3 = √2 × (r) × cos(θ)
    ------------------------------
    Z_4 = √2 × (r^2) × sin(2θ)
    Z_5 = 2r^2 + 1
    Z_6 = √2 × (r^2) × cos(2θ)
    ------------------------------
    Z_7 = √2 × (r^3) × sin(3θ)
    Z_8 = √2 × (3r^3 + -2r) × sin(θ)
    Z_9 = √2 × (3r^3 + -2r) × cos(θ)
    Z_10 = √2 × (r^3) × cos(3θ)
    ------------------------------
    Z_11 = √2 × (r^4) × sin(4θ)
    Z_12 = √2 × (4r^4 + -3r^2) × sin(2θ)
    Z_13 = 6r^4 + -6r^2 + 1
    Z_14 = √2 × (4r^4 + -3r^2) × cos(2θ)
    Z_15 = √2 × (r^4) × cos(4θ)
    ------------------------------"""
    def __init__(self, s, z_order, verbose=False, latex_output=False, orthonormal=False, check_orthogonal=False):
        self.s = s
        self.z_order = z_order
        self.verbose = verbose
        self.latex_output = latex_output
        self.orthonormal = orthonormal
        self.check_orthogonal = check_orthogonal
        self.r, self.theta, self.mask = self.get_polar_coordinates()
        self.zernike_polys = self.generate_zern_polynomials()
        self.nz = int((z_order + 1) * (z_order + 2) / 2)

        if self.check_orthogonal:
            self.check_orthogonality()

    def generate_algebraic_form(self, R_algebraic, m):
        if self.latex_output:
            return f"\\sqrt{{2}} \\times ({R_algebraic}) \\times \\sin({abs(m)}\\theta)" if m < 0 else \
                   f"\\sqrt{{2}} \\times ({R_algebraic}) \\times \\cos({m}\\theta)" if m > 0 else \
                   f"{R_algebraic}"
        else:
            return f"√2 × ({R_algebraic}) × sin({'' if abs(m) == 1 else abs(m)}θ)" if m < 0 else \
                   f"√2 × ({R_algebraic}) × cos({'' if m == 1 else m}θ)" if m > 0 else \
                   f"{R_algebraic}"

    def get_polar_coordinates(self):
        x = np.linspace(-1 / 2, 1 / 2, self.s)
        y = np.linspace(-1 / 2, 1 / 2, self.s)
        xx, yy = np.meshgrid(x, y)
        r = np.hypot(xx, yy)
        theta = np.arctan2(yy, xx)
        mask = (r <= 1 / 2)
        self.mask = mask
        return r, theta, mask

    def rms(self, array):
        return np.sqrt(np.mean(np.square(array)))

    def scale_to_rms(self, array, rms_value):
        current_rms = zernike.rms(array)
        target_rms = rms_value
        scale_factor = target_rms / current_rms
        array *= scale_factor
        return array

    def zernike_radial(self, n, m, rho):
        radial = 0
        algebraic_form = ""
        for k in range((n - abs(m)) // 2 + 1):
            coef = (-1) ** k * factorial(n - k) / (factorial(k) * factorial((n + abs(m)) // 2 - k) * factorial((n - abs(m)) // 2 - k))
            radial += coef * rho ** (n - 2 * k)
            factor = int(coef)
            if factor == 1:
                factor_str = ''
            else:
                factor_str = int(coef)
            exponent = n - 2 * k
            term = "1" if exponent == 0 else f"{factor_str}r" if exponent == 1 else f"{factor_str}r^{exponent}"
            algebraic_form += (" + " if k > 0 else "") + term
        return radial, algebraic_form

    def generate_zern_polynomials(self):
        z_polynomials = {}
        n, idx = 0, 0
        while n <= self.z_order:
            for m in range(-n, n + 1, 2):
                R, R_algebraic = self.zernike_radial(n, m, self.r)
                Z = np.sqrt(2) * R if m != 0 else R
                Z *= np.sin(-m * self.theta) if m < 0 else np.cos(m * self.theta) if m > 0 else 1
                if self.verbose:
                    algebraic_form = self.generate_algebraic_form(R_algebraic, m)
                    print(f"Z_{idx + 1} = {algebraic_form}")
                if self.orthonormal:
                    Z /= np.sqrt(np.sum(Z ** 2 * self.mask))
                z_polynomials[idx] = Z
                idx += 1
            if self.verbose:
                print('-'*30)
            n += 1
        return self.gram_schmidt(z_polynomials, self.mask)

    def check_orthogonality(self):
        # Check orthogonality or orthonormality here
        n = len(self.zernike_polys)
        othogonality_array = np.array([[round(np.sum(self.zernike_polys[i] * np.conj(self.zernike_polys[j]) * self.mask))
                                        for j in range(n)] for i in range(n)])
        print('-' * 30)
        print("Inner Products Table:")
        print('-' * 30)
        print(np.round(othogonality_array))
        return othogonality_array

    def sum_zern_coeff(self, z_coeff):
        wavefront = np.zeros_like(next(iter(self.zernike_polys.values())))
        for idx, c in enumerate(z_coeff):
            wavefront += c * self.zernike_polys[idx]
        return wavefront

    def plot_wavefront(self, final_wavefront):
        final_wavefront_masked = np.where(self.mask, final_wavefront, np.nan)
        cmap = plt.get_cmap('jet')
        cmap.set_bad(color='black')
        plt.figure()
        plt.imshow(final_wavefront_masked, cmap=cmap, extent=[-1, 1, -1, 1])
        plt.title('Final Wavefront')
        plt.colorbar(label='Amplitude')
        plt.tight_layout()

    def largest_prime_factor(self, n):
        i = 2
        while i * i <= n:
            n = n // i if n % i == 0 else n
            i = i + 1 if n % i != 0 else i
        return n

    def plot_zernike_terms(self):
        nz = len(self.zernike_polys)
        ncols = self.largest_prime_factor(nz)
        nrows = nz // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 4))
        axes = axes.flatten()
        for idx, ax in enumerate(axes[:nz]):
            zernike_term = self.zernike_polys[idx]
            zernike_term_masked = np.where(self.mask, zernike_term, np.nan)
            ax.imshow(zernike_term_masked, cmap='jet', extent=[-1, 1, -1, 1])
            ax.set_title(f'Z_{idx + 1}')
            ax.set_xticks([])  # Turn off x-axis ticks
            ax.set_yticks([])  # Turn off y-axis ticks
        for ax in axes[nz:]:
            ax.axis('off')
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()

    def inner_product(self, f, g, mask):
        return np.sum(f * np.conj(g) * mask)

    def gram_schmidt(self, zernike_polys, mask):
        # The Gram-Schmidt orthogonalization process is employed here to ensure that the generated
        # Zernike polynomials are numerically orthogonal over the unit disk. While Zernike polynomials
        # are theoretically orthogonal, numerical approximations can introduce errors. This process
        # corrects for such errors, making the code more robust and reliable for various applications.
        orthogonal_polys = {}
        for i, f in zernike_polys.items():
            new_f = f - sum(self.inner_product(f, g, mask) / self.inner_product(g, g, mask) * g for j, g in orthogonal_polys.items())
            orthogonal_polys[i] = new_f
        return orthogonal_polys


    def calculate_and_print_rms(self, z_coeff, final_wavefront):
        rms_coeff = np.sqrt(np.mean(z_coeff ** 2))
        rms_wf = np.sqrt(np.mean(final_wavefront ** 2))
        ratio = round(rms_wf / rms_coeff, 2)
        print(f"RMS(coeff) = {format_number(rms_coeff)}")
        print(f"RMS(WF) = {format_number(rms_wf)}")
        print(f"Ratio of the above = {ratio}; should be close to 1.")
        print("If not equal then perhaps there are numerical errors ")
        print("due to finite grid size or other grid alignment issue.")
        print('-' * 30)


if __name__ == "__main__":

    # Define parameters
    s = 128
    z_order = 4
    m_per_nm = 1e-9  # meters / nm
    wavelength = 550e-9
    f_num = 4.0
    use_dft = True
    check_orthogonal_basis = False
    defocus_waves = 10

    # Calculate the output size based on F/# and pupil size
    N = int(s * f_num)
    padsize = int((N - s) // 2)

    # Get Zernikes and wavefront
    zernike = ZernikePolynomials(s, z_order, verbose=True, latex_output=False, orthonormal=False, check_orthogonal=True)
    z_coeff = np.random.rand(zernike.nz) * wavelength/0.5
    z_coeff[0] = 0  # zero out piston
    # z_coeff = zernike.scale_to_rms(z_coeff, 1 * 2*np.pi/wavelength)
    wavefront = zernike.sum_zern_coeff(z_coeff) + defocus_waves*zernike.zernike_polys[4]*wavelength
    # final_wavefront = zernike.scale_to_rms(wavefront, 1 * 2*np.pi/wavelength)

    # Compare values of rms(coeff) amd rms(wavefront)
    zernike.calculate_and_print_rms(z_coeff, wavefront)

    mask = zernike.mask
    pupil_ca = mask * np.exp(1j * 2 * np.pi * wavefront/wavelength)

    # zernike.plot_wavefront(final_wavefront)
    # zernike.plot_zernike_terms()

    # Perform DFT or FFT to calculate the PSF amplitude
    if use_dft:
        psf_complex_amp = np.fft.fftshift(dft2d(pupil_ca, N))
    else:
        padded_pupil = np.pad(pupil_ca, ((padsize, padsize), (padsize, padsize)), mode='constant', constant_values=(0 + 0j))
        psf_complex_amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_pupil)))

    # Normalize the PSF
    irradiance = np.abs(psf_complex_amp) ** 2
    irradiance = irradiance / np.sum(irradiance)

    # Display results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # PSF
    im0 = axs[0].imshow(irradiance ** (1 / 2), cmap='gray')
    axs[0].set_title('Point Spread Function')
    plt.colorbar(im0, ax=axs[0], label='Intensity')

    # Wavefront
    im1 = axs[1].imshow(np.angle(pupil_ca), cmap='jet')
    axs[1].set_title('Wavefront')
    plt.colorbar(im1, ax=axs[1], label='Phase (radians)')

    # Amplitude
    im2 = axs[2].imshow(np.abs(pupil_ca), cmap='gray')
    axs[2].set_title('Amplitude')
    plt.colorbar(im2, ax=axs[2], label='Amplitude')

    plt.show()
