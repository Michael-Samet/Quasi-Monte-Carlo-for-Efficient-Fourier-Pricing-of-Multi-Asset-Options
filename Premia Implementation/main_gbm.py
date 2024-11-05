import numpy as np
from pricing_engine import QMC_fourier_pricing_engine
from models import covariance_matrix


def main():
    model_name = 'GBM'
    payoff_name = 'basket_put'
    
    # Define parameters
    d = 4
    S0 = 100 * np.ones(d)
    K = 100
    r = 0.1
    T = 1
    q = 0.05
    sigma = 0.2 * np.ones(d)
    rho = np.identity(d)
    SIGMA = covariance_matrix(sigma, rho)
    GBM_option_params = (d, S0, K, r, T, q, SIGMA)
    weights = np.ones(d) / d
    
    N_samples = 2**10
    S_shifts = 30
    
    price_estimate, error_estimate = QMC_fourier_pricing_engine(
        model_name,
        payoff_name,
        GBM_option_params,
        N_samples,
        S_shifts,
        transform_distribution= None,
        transform_params= None,
        weights = weights
    )

    print(f"Estimated Price: {round(price_estimate,5)}, Statistical Error: {round(error_estimate,5)}, Relative Error: {round(error_estimate / price_estimate,5)}")
if __name__ == "__main__":
    main()