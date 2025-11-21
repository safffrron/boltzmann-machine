# Introduction

## Motivation

Energy-Based Models (EBMs) represent a powerful and flexible framework for unsupervised learning, offering the ability to model complex data distributions without explicit density estimation. Unlike autoregressive or flow-based models, EBMs parameterize the unnormalized log-probability of data through an energy function, providing greater modeling flexibility at the cost of more challenging training procedures.

A critical component of EBM training is the sampling procedure used to estimate the gradient of the log-partition function. Contrastive Divergence (CD), introduced by Hinton (2002), revolutionized the practical training of EBMs by approximating the intractable negative phase gradient using short Markov Chain Monte Carlo (MCMC) chains. However, the number of MCMC steps (k in CD-k) introduces a fundamental trade-off: more steps yield better gradient estimates but increase computational cost, while fewer steps are efficient but may introduce bias.

Despite CD's widespread adoption, several key questions remain underexplored:

1. **Quality vs Computation Trade-off**: How does sample quality scale with the number of CD steps across different model architectures?
2. **CD vs PCD**: Does Persistent Contrastive Divergence (PCD), which maintains persistent MCMC chains across training iterations, offer advantages over standard CD?
3. **Architectural Dependence**: Do these trade-offs differ between simple models (RBMs) and complex deep architectures (Convolutional EBMs)?
4. **MCMC Mixing**: How do different CD-k values affect the mixing properties of the sampling chains?

## Research Questions

This project investigates these questions through systematic experimentation on two complementary model classes:

**RQ1**: How does the number of Contrastive Divergence steps (k) affect sample quality in Restricted Boltzmann Machines?

**RQ2**: What is the practical trade-off between computational cost and model performance for different CD-k values?

**RQ3**: How do CD and PCD compare in terms of sample quality, training stability, and MCMC mixing efficiency?

**RQ4**: Are the observations from simple models (RBMs) generalizable to modern deep energy-based architectures (Convolutional EBMs)?

**RQ5**: What hyperparameters (step size, noise, buffer size) most significantly impact the CD/PCD trade-off?

## Contributions

This work makes the following contributions:

1. **Comprehensive Empirical Study**: We provide a systematic comparison of CD-k for k ∈ {1, 5, 10, 20} across both RBMs and Convolutional EBMs, measuring sample quality through multiple metrics (FID, Inception Score, LPIPS diversity, reconstruction error, log-likelihood).

2. **CD vs PCD Analysis**: We directly compare standard CD against PCD across multiple k values, providing insights into when persistent chains offer advantages and quantifying the memory-performance trade-off.

3. **MCMC Diagnostics**: Beyond sample quality, we analyze the MCMC dynamics through autocorrelation analysis, effective sample size computation, and mixing time estimation, providing deeper understanding of the sampling process.

4. **Practical Guidelines**: We derive practical recommendations for practitioners choosing CD-k values, including compute-quality trade-off curves and architecture-specific guidelines.

5. **Reproducible Implementation**: We provide a complete, well-documented codebase optimized for commodity hardware (Kaggle GPUs), making our experiments easily reproducible and extensible.

## Scope and Limitations

**Scope**:
- Binary RBMs on MNIST/FashionMNIST (28×28 grayscale)
- Convolutional EBMs on CIFAR-10 (32×32 RGB)
- CD-k for k ∈ {1, 5, 10, 20}
- PCD variants with replay buffers
- Langevin dynamics for continuous EBMs
- Multiple evaluation metrics and MCMC diagnostics

**Limitations**:
- Models are intentionally kept small (~1-4M parameters) for tractability on Kaggle GPUs
- Limited to relatively simple datasets (MNIST, CIFAR-10)
- Focus on image generation; other modalities not explored
- Training limited to 30-50 epochs due to computational constraints
- Ablation studies focus on key hyperparameters; exhaustive search not performed

Despite these limitations, our findings provide valuable insights into the CD-k trade-off that generalize to larger-scale settings.

## Organization

The remainder of this report is organized as follows:

- **Section 2 (Background)**: Reviews Energy-Based Models, Contrastive Divergence, MCMC sampling, and related work
- **Section 3 (Methods)**: Details our model architectures, training procedures, and evaluation metrics
- **Section 4 (Experiments)**: Describes experimental setup, datasets, and configurations
- **Section 5 (Results)**: Presents quantitative and qualitative results from all experiments
- **Section 6 (Analysis)**: Analyzes findings, discusses trade-offs, and derives insights
- **Section 7 (Conclusion)**: Summarizes contributions and suggests future directions
- **Appendix**: Provides implementation details, hyperparameters, and additional results

## Expected Impact

This work aims to:

1. **Guide Practitioners**: Provide evidence-based recommendations for choosing CD-k values in practical applications
2. **Inform Architecture Design**: Highlight how architectural choices interact with sampling strategies
3. **Advance Understanding**: Deepen our understanding of MCMC sampling in energy-based learning
4. **Enable Future Research**: Provide a solid foundation and reproducible baseline for future investigations

By systematically investigating the role of CD steps on sample quality, we hope to contribute to more efficient and effective training of Energy-Based Models, ultimately advancing the state of generative modeling.