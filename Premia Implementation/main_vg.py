import numpy as np
from pricing_engine import QMC_fourier_pricing_engine
from models import covariance_matrix

import warnings
warnings.filterwarnings("ignore")


def main():
    model_name = 'VG'
    payoff_name = 'cash_or_nothing_put'
    
    # Define parameters
    d = 2
    S0 = 100 * np.ones(d)
    K = 100
    r = 0
    q = 0
    T = 1
    nu = 0.001
    sigma = 0.4 * np.ones(d)
    theta = -0.3 * np.ones(d)
    rho = np.identity(d)
    SIGMA = covariance_matrix(sigma, rho)
    print(SIGMA)
    TOLR = None # specify relative error
    VG_option_params = (d,S0,K,r,q,T,SIGMA,theta,nu)
    N_samples = 2**7
    S_shifts = 30
    
    price_estimate, error_estimate = QMC_fourier_pricing_engine(
        model_name,
        payoff_name,
        VG_option_params,
        N_samples,
        S_shifts,
        transform_distribution= None,
        transform_params= None,
        TOLR = TOLR
    )

    print(f"Estimated Price: {round(price_estimate,5)}, Statistical Error: {round(error_estimate,5)}, Relative Error: {round(error_estimate / price_estimate,5)}")

if __name__ == "__main__":
    main()