import numpy as np
import scipy.linalg as la
from scipy.stats import rayleigh
from scipy.special import kv
from scipy.stats import multivariate_t, chi
from scipy.stats import multivariate_normal

class MLTransform:
    def __init__(self, SIGMA_IS):
        self.SIGMA_IS = SIGMA_IS
        self.SIGMA_IS_inv = la.inv(SIGMA_IS)

    def pdf(self, u):
        return multivariate_laplace_pdf(u, self.SIGMA_IS, self.SIGMA_IS_inv)

    def icdf(self, u):
        return rayleigh.ppf(q=u, loc=0, scale=1/np.sqrt(2))

    def decompose(self, method='cholesky'):
        if method == 'cholesky':
            return la.cholesky(self.SIGMA_IS)
        elif method == 'EVD':
            eigvals, P = la.eig(self.SIGMA_IS)
            indices = np.argsort(eigvals)[::-1]
            eigvals_sorted = eigvals[indices]
            P_sorted = P[:, indices]
            return P_sorted @ np.diag(np.sqrt(eigvals_sorted))
        else:
            raise ValueError("Unsupported decomposition method")

def multivariate_laplace_pdf(x, SIGMA_IS, SIGMA_IS_inv):
    d = len(x)
    v = (2 - d) / 2
    f_ML = 2 * (2 * np.pi) ** (-d / 2) * (la.det(SIGMA_IS)) ** (-1/2) * (x @ SIGMA_IS_inv @ x / 2) ** (v / 2) * kv(v, np.sqrt(2 * x @ SIGMA_IS_inv @ x))
    return f_ML

class MSTransform:
    def __init__(self, SIGMA_IS, nu_IS):
        self.SIGMA_IS = SIGMA_IS
        self.nu_IS = nu_IS

    def pdf(self, u):
        """Compute the PDF of the multivariate t-distribution."""
        return multivariate_t.pdf(x=u, df=self.nu_IS, loc=np.zeros(len(u)), shape=self.SIGMA_IS)

    def icdf(self, u):
        """Compute the inverse CDF for the transformation."""
        return 1 / chi.ppf(q=u, df=self.nu_IS, loc=0, scale=1)

    def decompose(self, method='cholesky'):
        """Decompose the covariance matrix using the specified method."""
        if method == 'cholesky':
            return la.cholesky(self.nu_IS * self.SIGMA_IS)
        elif method == 'EVD':
            eigvals, P = la.eig(self.nu_IS * self.SIGMA_IS)
            indices = np.argsort(eigvals)[::-1]
            eigvals_sorted = eigvals[indices]
            P_sorted = P[:, indices]
            return P_sorted @ np.diag(np.sqrt(eigvals_sorted))
        else:
            raise ValueError("Unsupported decomposition method")
        
        
class MNTransform:
    def __init__(self, SIGMA_IS):
        self.SIGMA_IS = SIGMA_IS

    def pdf(self, u):
        """Compute the PDF of the multivariate normal distribution."""
        return multivariate_normal.pdf(x=u, mean=np.zeros(len(u)), cov=self.SIGMA_IS)

    def icdf(self, u):
        """For normal distribution transformation, ICDF is identity."""
        return 1

    def decompose(self, method='cholesky'):
        """Decompose the covariance matrix using the specified method."""
        if method == 'cholesky':
            return la.cholesky(self.SIGMA_IS)
        elif method == 'EVD':
            eigvals, P = la.eig(self.SIGMA_IS)
            indices = np.argsort(eigvals)[::-1]
            eigvals_sorted = eigvals[indices]
            P_sorted = P[:, indices]
            return P_sorted @ np.diag(np.sqrt(eigvals_sorted))
        else:
            raise ValueError("Unsupported decomposition method")