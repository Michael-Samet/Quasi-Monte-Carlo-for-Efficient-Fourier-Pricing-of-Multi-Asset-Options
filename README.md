# Quasi-Monte-Carlo-for-Efficient-Fourier-Pricing-of-Multi-Asset-Options

This repository contains the implementation of the randomized quasi-Monte Carlo (RQMC) method applied in the Fourier space for the pricing of multi-asset European rainbow and basket options under Geometric Brownian motion (GBM), variance gamma (VG), and normal inverse Gaussian (NIG).  It also contains the implementation of the tensor product Gauss-Laguerre quadrature method and the Monte Carlo (MC) method applied in the Fourier domain. Moreover, we provide the implementation of the standard MC method in the physical/direct domain for benchmarking purposes and the computation of reference prices.

The implementation is in the Python programming language, and the presented functions are commented in detail. 
The notation in the code is consistent with the two following works:
- C. Bayer, C. Ben Hammouda, A. Papapantoleon, M. Samet, and R. Tempone. Optimal damping with hierarchical adaptive quadrature for efficient Fourier pricing of multi-asset options in Lévy models. Journal of Computational Finance, 27(3):43–86, 2024.
- C. Bayer, C. Ben Hammouda, A. Papapantoleon, M. Samet, and R. Tempone. Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options (To appear soon).
