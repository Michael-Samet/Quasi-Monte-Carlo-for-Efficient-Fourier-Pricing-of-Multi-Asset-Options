# Quasi-Monte-Carlo-for-Efficient-Fourier-Pricing-of-Multi-Asset-Options

This repository contains the implementation of the randomized quasi-Monte Carlo (RQMC) method applied in the Fourier space for the pricing of multi-asset European rainbow, basket, spread, and digital options under Geometric Brownian motion (GBM), variance gamma (VG), normal inverse Gaussian (NIG) and Generalized Hyperbolic (GH) models.  It also contains the implementation of the tensor product Gauss-Laguerre quadrature method and the Monte Carlo (MC) method applied in the Fourier domain. Moreover, we provide the implementation of the standard MC method in the physical/direct domain for benchmarking purposes and the computation of reference prices.

The Premia Implementation folder features an object-oriented programming structure developed during a research visit with the MathRisk team at Inria in Paris. This structure enhances modularity and facilitates easier extensions to other models and payoff functions. Comprehensive documentation is included to support understanding and further development.

The implementation is in the Python programming language, and the presented functions are commented in detail. 
The notation in the code is consistent with the two following works:
- [C. Bayer, C. Ben Hammouda, A. Papapantoleon, M. Samet, and R. Tempone. Optimal damping with hierarchical adaptive quadrature for efficient Fourier pricing of multi-asset options in Lévy models. Journal of Computational Finance, 27(3):43–86, 2024.](https://arxiv.org/abs/2203.08196)
- [C. Bayer, C. Ben Hammouda, A. Papapantoleon, M. Samet, and R. Tempone. Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options (Arxiv preprint)](https://arxiv.org/abs/2403.02832)


In case of use of the code, please cite us by mentioning the webpage containing the code and adding the following references to your work:

@article{bayer2022optimal,
  title={Optimal Damping with Hierarchical Adaptive Quadrature for Efficient Fourier Pricing of Multi-Asset Options in L$\backslash$'evy Models},
  author={Bayer, Christian and Hammouda, Chiheb Ben and Papapantoleon, Antonis and Samet, Michael and Tempone, Ra{\'u}l},
  journal={Journal of Computational Finance 27.3 (2024), pp. 43–86},
  year={2024}
}

@article{bayer2024quasi,
  title={Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options},
  author={Bayer, Christian and Hammouda, Chiheb Ben and Papapantoleon, Antonis and Samet, Michael and Tempone, Ra{\'u}l},
  journal={arXiv preprint arXiv:2403.02832},
  year={2024}
}
