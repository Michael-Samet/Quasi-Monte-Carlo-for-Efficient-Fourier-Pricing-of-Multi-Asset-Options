import numpy as np
from scipy.special import kv

class NIGModel:
    """Normal Inverse Gaussian (NIG) model 
    
    This class implements the NIG model. The model is parameterized by alpha, beta, 
    delta, and DELTA, which control the shape of the distribution.

    Attributes:
        d (int):  Number of assets.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        alpha (float): Tail/steepness parameter controlling kurtosis.
        beta (np.ndarray): Skewness parameters.
        delta (float): Scale parameter analogous to volatility in other models.
        DELTA (np.ndarray):  Matrix controlling the correlation between assets.
        mu (float): Martingale correction term to ensure no arbitrage.
    """
    
    def __init__(self, d: int, T: float, r: float, q: float,
                 alpha: float, beta: float, delta: float, DELTA: np.ndarray):
        """Initializes an instance of the NIGModel class.

        Args:
            d (int): Dimensionality of the model.
            T (float): Time to maturity.
            r (float): Risk-free interest rate.
            q (float): Dividend yield.
            alpha (float): Tail/steepness parameter controlling kurtosis.
            beta (np.ndarray): Skewness parameters.
            delta (float): Scale parameter analogous to volatility in other models.
            DELTA (np.ndarray): Covariance matrix for the model.

        """
        self.d = d
        self.T = T
        self.r = r
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.DELTA = DELTA
        
        # Compute the martingale correction term to ensure the model is arbitrage-free
        self.mu = self.compute_mu()
        
    def compute_mu(self) -> float:
        """Computes the martingale correction term `mu`.
        
        The martingale correction ensures that the discounted price process is a 
        martingale under the risk-neutral measure. This term adjusts for drift 
        to prevent arbitrage opportunities.

        Returns:
            float: The computed value of `mu`.
        
        """
        # Martingale correction formula derived from NIG dynamics
        return -self.delta * (
            np.sqrt(self.alpha**2 - np.square(self.beta)) - 
            np.sqrt(self.alpha**2 - np.square(self.beta + 1))
        )
    
    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """Computes the characteristic function of the NIG model.

        The characteristic function is a key component in option pricing and 
        other financial applications. It represents the Fourier transform of 
        the probability distribution of returns.

        Args:
            u (np.ndarray): Complex argument for which to evaluate the characteristic function.

        Returns:
            np.ndarray: The value of the characteristic function at `u`.

        """
        
        # Exponent term involving drift adjustment and time scaling
        drift_term = 1j * self.T * (self.r - self.q + self.mu) @ u
        
        # Volatility term involving delta and covariance matrix DELTA
        volatility_term = self.delta * self.T * (
            np.sqrt(self.alpha**2 - self.beta @ self.DELTA @ self.beta) - 
            np.sqrt(self.alpha**2 - (self.beta + 1j * u) @ self.DELTA @ (self.beta + 1j * u))
        )
        
        # Characteristic function formula combining drift and volatility terms
        phi = np.exp(drift_term + volatility_term)
        
        return phi
    
    def strip_of_analyticity(self):
        """Defines the strip of analyticity.

        Returns:
            Callable[[float], float]: A lambda function that evaluates whether `R` belongs to 
                                      the strip of analyticity based on model parameters.
        
        """
        
        # Lambda function that checks whether R lies within the strip of analyticity
        return lambda R: (
            self.alpha**2 - (self.beta - R) @ self.DELTA @ (self.beta - R)
        )
        
    

class GHModel:
    """Generalized Hyperbolic (GH) model.
    
    This class implements the GH model. The model is parameterized by alpha, beta, 
    delta, DELTA, and lambda_var, which control the shape of the distribution.

    Attributes:
        d (int): Number of assets.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        alpha (float): Tail/steepness parameter controlling kurtosis.
        beta (np.ndarray): Skewness parameter vector.
        delta (float): Scale parameter analogous to volatility in other models.
        DELTA (np.ndarray):  Matrix controlling the correlation between assets.
        lambda_var (float): Shape parameter controlling the GH distribution's tail behavior.
        mu_GH (np.ndarray): Martingale correction term to ensure no arbitrage.
    """
    
    def __init__(self, d: int, T: float, r: float, q: float,
                 alpha: float, beta: np.ndarray, delta: float,
                 DELTA: np.ndarray, lambda_var: float):
        """Initializes an instance of the GHModel class.

        Args:
            d (int): Number of assets.
            T (float): Time to maturity
            r (float): Risk-free interest rate.
            q (float): Dividend yield.
            alpha (float): Tail/steepness parameter controlling kurtosis.
            beta (np.ndarray): Skewness parameter vector of length `d`.
            delta (float): Scale parameter analogous to volatility in other models.
            DELTA (np.ndarray): Matrix controlling the correlation between assets
            lambda_var (float): Shape parameter controlling tail behavior in GH distribution.

        """
        self.d = d
        self.T = T
        self.r = r
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.DELTA = DELTA
        self.lambda_var = lambda_var
        
        # Compute the martingale correction term to ensure no arbitrage
        self.mu_GH = self.compute_mu()

    def compute_mu(self) -> np.ndarray:
        """Computes the martingale correction term `mu_GH`.
        
        The martingale correction ensures that the discounted price process is a 
        martingale under the real-world measure. This term adjusts for drift 
        to prevent arbitrage opportunities.

        Returns:
            np.ndarray: The computed value of `mu_GH`, a vector of length `d`.

        """
        
        # Initialize mu_GH as a zero vector of dimension `d`
        mu_GH = np.zeros(self.d)
        
        # Loop over each dimension to compute mu_GH[i] using real-world characteristic function
        for i in range(self.d):
            beta_i = self.beta[i]
            
            # Martingale correction formula derived from GH dynamics
            mu_GH[i] = -1 / self.T * np.log(self.real_world_characteristic_function(-1j, beta_i))
        
        return mu_GH

    def real_world_characteristic_function(self, u: complex, beta_i: float) -> complex:
        """Computes the real-world characteristic function for a given `u` and `beta_i`.
        
        This function is used in calculating the martingale correction term `mu_GH`. It 
        evaluates the characteristic function under the real-world measure.

        Args:
            u (complex): Complex argument for which to evaluate the characteristic function.
            beta_i (float): The `i`-th element of the skewness vector `beta`.

        Returns:
            complex: The value of the real-world characteristic function at `u` and `beta_i`.

        """
        
        # First part of characteristic function involving ratio of squared terms
        phi = ((self.alpha**2 - beta_i**2) / 
               (self.alpha**2 - (beta_i + 1j * u)**2))**(self.lambda_var / 2)
        
        # Bessel function terms involving modified Bessel functions of second kind (`kv`)
        phi *= kv(self.lambda_var, self.T * self.delta * np.sqrt(self.alpha**2 - (beta_i + 1j * u)**2))
        
        # Normalize by dividing by Bessel function evaluated at zero argument
        phi /= kv(self.lambda_var, self.T * self.delta * np.sqrt(self.alpha**2 - beta_i**2))
        
        return phi

    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """Computes the characteristic function of the GH model.

        Args:
            u (np.ndarray): Complex argument for which to evaluate the characteristic function.

        Returns:
            np.ndarray: The value of the characteristic function at `u`.

        """
        
        # Exponent term involving drift adjustment and time scaling
        drift_term = 1j * self.T * (self.r - self.q + self.mu_GH) @ u
        
        # Volatility term involving delta and covariance matrix DELTA
        volatility_term = ((self.alpha**2 - self.beta @ self.DELTA @ self.beta) /
                           (self.alpha**2 - self.beta @ self.DELTA @ self.beta + u @ self.DELTA @ u - 2j * self.beta @ self.DELTA @ u))**(self.lambda_var / 2)
        
        # Bessel function terms involving modified Bessel functions of second kind (`kv`)
        bessel_term = kv(self.lambda_var, 
                         self.T * self.delta * np.sqrt(self.alpha**2 - 
                                                       self.beta @ self.DELTA @ self.beta + 
                                                       u @ self.DELTA @ u - 2j * self.beta @ self.DELTA @ u))
        
        # Normalize by dividing by Bessel function evaluated at zero argument
        normalization_term = kv(self.lambda_var,
                                self.T * self.delta * np.sqrt(self.alpha**2 - 
                                                              self.beta @ self.DELTA @ self.beta))
        
        # Combine all terms to compute final characteristic function value
        phi = np.exp(drift_term) * volatility_term * bessel_term / normalization_term
        
        return phi
    
    def strip_of_analyticity(self):
        """Defines a lambda function representing the strip of analyticity.

        Returns:
            Callable[[float], float]: A lambda function that evaluates whether `R` belongs to 
                                      the strip of analyticity based on model parameters.

         """
         
         # Lambda function that checks whether R lies within the strip of analyticity
        return lambda R: (
             self.alpha**2 - (self.beta - R) @ self.DELTA @ (self.beta - R)
         )

class VGModel:
    """Variance Gamma (VG) model.
    
    The VG model is parameterized by SIGMA (covariance matrix), theta (drift), and nu 
    (variance of the subordinator).

    Attributes:
        d (int): Number of Assets
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        SIGMA (np.ndarray): Covariance matrix
        theta (np.ndarray): Drift vector.
        nu (float): Variance of the Gamma process subordinator.
        mu (np.ndarray): Martingale correction term to ensure no arbitrage.
    """
    
    def __init__(self, d: int, T: float, r: float, q: float,
                 SIGMA: np.ndarray, theta: np.ndarray, nu: float):
        """Initializes an instance of the VGModel class.

        Args:
            d (int): Number of Assets
            T (float): Time to maturity.
            r (float): Risk-free interest rate.
            q (float): Dividend yield.
            SIGMA (np.ndarray): Covariance matrix.
            theta (np.ndarray): Drift vector.
            nu (float): Variance of the Gamma process subordinator.

        """
        self.d = d
        self.T = T
        self.q = q
        self.r = r
        self.SIGMA = SIGMA
        self.theta = theta
        self.nu = nu
        
        # Compute the martingale correction term
        self.mu = self.compute_mu()

    def compute_mu(self) -> np.ndarray:
        """Computes the martingale correction term `mu`.
        
        This term ensures that the discounted price process is a martingale 
        under the risk-neutral measure.

        Returns:
            np.ndarray: The computed value of `mu`, a vector of length `d`.

        """
        
        # Martingale correction formula for VG dynamics
        return (1 / self.nu) * np.log(1 - self.nu * self.theta - 0.5 * self.nu * np.diag(self.SIGMA))

    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """Computes the characteristic function of the VG model.
        Args:
            u (np.ndarray): Complex argument for which to evaluate the characteristic function.

        Returns:
            np.ndarray: The value of the characteristic function at `u`.

        """
        
        # Exponent term involving drift adjustment and time scaling
        drift_term = np.exp(1j * self.T * np.dot(self.r - self.q + self.mu, u))
        
        # Volatility term involving theta and SIGMA
        volatility_term = (
            1 - 1j * self.nu * np.dot(self.theta, u) +
            0.5 * self.nu * np.dot(u, np.dot(self.SIGMA, u))
        ) ** (-self.T / self.nu)
        
        # Combine drift and volatility terms to compute characteristic function
        phi = drift_term * volatility_term
        
        return phi

    def strip_of_analyticity(self):
        """Defines a lambda function representing the strip of analyticity.

        Returns:
            Callable[[np.ndarray], float]: A lambda function that evaluates whether 
                                           a given real vector `R` satisfies the 
                                           strip of analyticity constraint.

         """
         
         # Lambda function defining the constraint for strip of analyticity
        return lambda R: 1 + self.nu * np.dot(self.theta, R) - 0.5 * self.nu * np.dot(R, np.dot(self.SIGMA, R))
    


class GBMModel:
    """Geometric Brownian Motion (GBM)model.
    
    The GBM model is parameterized by SIGMA (covariance matrix)

    Attributes:
        d (int): Number of assets in the model.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        SIGMA (np.ndarray): Covariance matrix
    """
    
    def __init__(self, d: int, T: float, r: float, q: float, SIGMA: np.ndarray):
        """Initializes an instance of the GBMModel class.

        Args:
            d (int): Number of assets in the model.
            T (float): Time to maturity.
            r (float): Risk-free interest rate.
            q (float): Dividend yield.
            SIGMA (np.ndarray): Covariance matrix 

        """
        self.d = d
        self.T = T
        self.r = r
        self.q = q
        self.SIGMA = SIGMA

    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """Computes the characteristic function of the GBM model.

        Args:
            u (np.ndarray): Complex argument for which to evaluate the characteristic function.

        Returns:
            np.ndarray: The value of the characteristic function at `u`.

        """
        
        # Compute drift term involving risk-free rate, dividend yield, and volatility adjustment
        drift_term = np.exp(np.dot(1j * self.T * u, self.r - self.q - 0.5 * np.diag(self.SIGMA)))
        
        # Compute volatility term involving covariance matrix SIGMA
        volatility_term = np.exp(-0.5 * self.T * np.dot(u, np.dot(self.SIGMA, u)))
        
        # Combine drift and volatility terms to compute characteristic function
        phi = drift_term * volatility_term
        
        return phi

    def strip_of_analyticity(self):
        """Defines a non-restrictive constraint for the GBM model's strip of analyticity.
        
        Returns:
            Callable[[np.ndarray], int]: A lambda function that always returns 0.

         """
         
         # Non-restrictive constraint for GBM's strip of analyticity
        return lambda R: 0
    
    
def covariance_matrix(sigma, rho):
    """Compute the covariance matrix."""
    sigma = np.diag(sigma)  # Diagonal matrix of volatilities
    SIGMA = np.dot(sigma, np.dot(rho, sigma))  # Covariance matrix calculation
    return SIGMA