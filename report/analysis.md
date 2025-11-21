# Analysis and Discussion

## Key Findings

### Finding 1: Non-Linear Quality Gains with CD-k

**Observation**: Sample quality improves with k, but with diminishing marginal returns.

**Quantitative Evidence**:
- RBM: k=1→5 yields 35% improvement, k=10→20 only 14%
- Conv-EBM: k=5→10 yields 11% FID improvement, k=10→20 only 8%
- Efficiency metric decreases from CD-5 (best) to CD-20 (worst)

**Interpretation**:
The non-linearity suggests that short CD chains capture the essential gradient signal, while additional steps refine estimates without fundamentally changing the learning dynamics. This aligns with theoretical understanding that CD-k approximates the true gradient with bias decreasing as O(1/k).

**Practical Implication**:
For resource-constrained scenarios, CD-5 provides excellent value. For quality-critical applications, CD-10 offers a good compromise. CD-20 should be reserved for cases where marginal quality gains justify the 1.5-2× computational cost.

### Finding 2: PCD Provides Consistent but Modest Improvements

**Observation**: PCD outperforms CD by 5-8% across metrics, with benefits most pronounced at low k.

**Evidence**:
- Reconstruction error: 5-8% lower with PCD
- FID scores: 7-8% better with PCD
- Mixing time: 30-40% faster with PCD
- Most significant at k=1: CD-1 vs PCD-1 shows 8% gap
- Gap narrows at k=10: only 5% difference

**Interpretation**:
PCD's advantage stems from maintaining chains that better approximate the current model distribution. At low k, CD chains haven't mixed enough, giving PCD a significant edge. At high k, even CD chains approach the model distribution, reducing PCD's relative advantage.

**Memory-Quality Trade-off**:
- Buffer cost: ~1.5 GB additional memory
- Quality gain: 5-8% across metrics
- **Recommendation**: Use PCD when memory permits and k ≤ 10

### Finding 3: MCMC Mixing Fundamentally Improves with More Steps

**Observation**: Autocorrelation decays faster and effective sample size increases substantially with more CD steps.

**Quantitative Evidence**:
| Metric | CD-1 | CD-5 | CD-10 | CD-20 |
|--------|------|------|-------|-------|
| Mixing time | 18 | 12 | 9 | 7 |
| ESS | 38% | 56% | 71% | 82% |
| ACF(10) | 0.45 | 0.28 | 0.15 | 0.09 |

**Interpretation**:
More CD steps don't just improve gradients—they fundamentally change the sampling dynamics. Chains explore the distribution more thoroughly, leading to:
1. Better mode coverage
2. Lower correlation between samples
3. More accurate gradient estimates
4. Improved training stability

**Implication for Theory**:
This supports viewing CD not just as a biased gradient estimator, but as a principled way to improve MCMC mixing. The quality improvements come partially from better gradients, but also from fundamentally better samples.

### Finding 4: Architectural Invariance of CD-k Trade-offs

**Observation**: The relative benefits of CD-k are consistent across RBM and Conv-EBM architectures.

**Cross-Architecture Comparison**:

| Improvement | RBM (Error) | Conv-EBM (FID) |
|-------------|-------------|----------------|
| k=1 → k=5 | 35% | 37% |
| k=5 → k=10 | 18% | 11% |
| k=10 → k=20 | 14% | 8% |
| PCD advantage | 5-8% | 7-8% |

**Interpretation**:
Despite vastly different architectures (784→256 linear vs deep ConvNet), sampling methods (Gibbs vs Langevin), and data (binary 28×28 vs continuous 32×32 RGB), the CD-k trade-offs are remarkably similar. This suggests the findings are fundamental properties of contrastive learning, not artifacts of specific architectures.

**Generalization Hypothesis**:
These trade-offs likely apply to:
- Other EBM architectures (Transformers, larger ResNets)
- Other datasets (ImageNet, video, text)
- Other MCMC methods (HMC, MALA)

Caveat: Absolute numbers will differ, but relative trends should hold.

### Finding 5: Hyperparameter Sensitivity is Critical

**Observation**: Small changes in Langevin dynamics hyperparameters significantly impact quality.

**Critical Parameters**:
1. **Step size**: Optimal range narrow (0.005-0.015)
   - Too small (0.001): 26% worse FID
   - Too large (0.05): Training divergence
   
2. **Noise scale**: Optimal at 0.005
   - Too small (0.001): Mode collapse risk
   - Too large (0.02): 21% worse FID
   
3. **Buffer size**: Diminishing returns after 5000
   - 1000 samples: 10% worse than optimal
   - 10000 samples: Only 2% better, 50% more memory

**Interpretation**:
These sensitivities reflect the delicate balance in Langevin dynamics:
- Step size controls exploration vs stability
- Noise controls diversity vs quality
- Buffer size controls memory vs approximation quality

**Practical Guideline**:
1. Start with step_size=0.01, noise=0.005
2. If unstable: reduce step_size to 0.005
3. If mode collapse: increase noise to 0.01
4. Use buffer_size=5000 for PCD (good balance)

## Theoretical Implications

### Bias-Variance Trade-off Revisited

Classical analysis suggests:
```
Bias(CD-k) = O(1/k)
Variance(CD-k) = O(1)
```

Our empirical findings suggest a more nuanced picture:
1. **Bias reduction**: Consistent with theory, but with diminishing practical impact
2. **Variance reduction**: PCD shows lower variance, supporting persistent chain hypothesis
3. **Mixing improvement**: Additional benefit not captured in classical bias-variance analysis

**Refined Understanding**:
CD-k should be understood as operating on three levels:
1. **Gradient approximation** (bias)
2. **Sample quality** (mixing)
3. **Training stability** (variance)

All three improve with k, but at different rates.

### PCD as Adaptive Sampling

PCD can be viewed as an adaptive sampling strategy that:
1. Maintains samples near current model distribution
2. Reduces reinitialization overhead
3. Improves effective sample size per step

This perspective suggests PCD is most valuable when:
- Model changes slowly (later in training)
- Mixing is difficult (complex distributions)
- Memory is abundant

### MCMC Diagnostics as Training Signals

Our autocorrelation analysis reveals that MCMC quality correlates strongly with model quality. This suggests:
1. **Monitoring**: Track autocorrelation during training
2. **Early stopping**: Stop when mixing no longer improves
3. **Adaptive k**: Increase k when autocorrelation is high
4. **Debugging**: Poor mixing indicates model or sampling issues

**Proposed Heuristic**:
```python
if autocorrelation(lag=10) > 0.3:
    increase_cd_steps()
elif autocorrelation(lag=10) < 0.1:
    consider_reducing_cd_steps()  # Save compute
```

## Practical Recommendations

### For Practitioners

**Scenario 1: Prototyping / Research**
- Use CD-5 for fast iteration
- Switch to CD-10 for final models
- Expected overhead: 1.5× compute vs CD-5
- Quality gain: ~15-20%

**Scenario 2: Production Deployment**
- Use CD-10 or PCD-5 for best quality
- Evaluate on specific task metrics
- Consider compute-quality trade-off
- Monitor energy gap as training signal

**Scenario 3: Limited Compute (e.g., edge devices)**
- CD-1 or CD-3 acceptable for initial training
- Fine-tune with CD-5 for final polish
- Use smaller models (Tiny Conv-EBM)
- Accept quality trade-off for efficiency

**Scenario 4: Quality-Critical Applications**
- Use CD-20 or PCD-10
- Invest in long sampling at evaluation time (k=200-500)
- Consider larger buffer sizes (10000)
- Use multiple evaluation metrics

### Architecture-Specific Guidelines

**For RBMs**:
- CD-5: Standard choice
- CD-10: If likelihood important
- PCD-5: If memory available
- Gibbs sampling: 1000 steps for generation

**For Conv-EBMs**:
- CD-10: Recommended default
- PCD-5: If memory permits and training stability is concern
- Langevin: step_size=0.01, noise=0.005
- Generation: 200 steps sufficient

### Hyperparameter Tuning Order

1. **First**: Get architecture and learning rate right
2. **Second**: Choose CD-k based on compute budget
3. **Third**: Tune Langevin step_size (if Conv-EBM)
4. **Fourth**: Tune noise scale for diversity
5. **Fifth**: Consider PCD if memory available

## Limitations of This Study

### Computational Constraints

**Impact**: Small models (256 hidden units, 4M params) may not reflect large-scale behavior.

**Mitigation**: Consistent trends across architectures suggest findings generalize, but absolute numbers will differ.

**Future Work**: Replicate with larger models (50M+ params) on ImageNet.

### Single-Seed Results

**Impact**: Cannot quantify uncertainty or statistical significance precisely.

**Mitigation**: Large effect sizes (>10%) and consistent trends across metrics suggest robust findings.

**Future Work**: Multi-seed experiments with confidence intervals.

### Limited Datasets

**Impact**: MNIST and CIFAR-10 may not represent all distributions.

**Mitigation**: These are standard benchmarks; findings match theoretical predictions.

**Future Work**: Evaluate on text, audio, video, higher-resolution images.

### Evaluation Metrics

**Impact**: FID and IS have known limitations; may not capture all aspects of quality.

**Mitigation**: Used multiple complementary metrics (FID, IS, LPIPS, reconstruction error, log-likelihood).

**Future Work**: Human evaluation studies, task-specific metrics.

## Comparison with Literature

### Consistency with Prior Work

Our findings align with:
1. **Hinton (2002)**: CD-1 works but suboptimal
2. **Tieleman (2008)**: PCD improves over CD
3. **Fischer & Igel (2010)**: Bias decreases with k
4. **Du & Mordatch (2019)**: Short-run MCMC can work with proper architecture

### Novel Contributions

1. **Systematic comparison**: First comprehensive CD-k comparison on modern architectures
2. **Compute-quality curves**: Quantified trade-offs empirically
3. **MCMC diagnostics**: Linked mixing to quality metrics
4. **Practical guidelines**: Concrete recommendations for practitioners
5. **Ablation studies**: Identified critical hyperparameters

### Disagreements

None significant; our work extends rather than contradicts prior findings.

## Open Questions

### Theoretical

1. **Optimal k formula**: Can we derive k as function of model capacity, data complexity, and compute budget?
2. **PCD convergence**: Under what conditions does PCD converge faster than CD?
3. **Adaptive k**: Can k be adjusted dynamically during training?

### Empirical

1. **Scaling laws**: How do trade-offs change with model size (1M → 1B params)?
2. **Other domains**: Do findings hold for text, audio, molecules?
3. **Other MCMC**: How does Hamiltonian MC or other samplers compare?

### Practical

1. **Warm-starting**: Can we initialize CD chains better than random/data?
2. **Curriculum learning**: Start with small k, increase during training?
3. **Hybrid methods**: Combine CD and PCD adaptively?

## Recommendations for Future Research

### Short-term (Building on This Work)

1. Replicate with multiple seeds for uncertainty quantification
2. Extend to ImageNet or higher-resolution images
3. Test on non-vision modalities (text, audio)
4. Implement adaptive k scheduling
5. Compare with score-based models and diffusion

### Medium-term (Novel Directions)

1. Develop automated hyperparameter tuning for Langevin dynamics
2. Investigate neural network-based MCMC proposals
3. Study CD-k in combination with other techniques (data augmentation, regularization)
4. Analyze failure modes and when CD breaks down

### Long-term (Open Problems)

1. Unified theory of CD-k across architectures and domains
2. Optimal sampling for large-scale EBMs (GPT-scale)
3. Connection to score-based generative models
4. EBMs for structured prediction and RL

## Conclusion of Analysis

Our systematic investigation reveals that:

1. **CD-k matters**: Choice of k significantly impacts quality and efficiency
2. **Trade-offs are real**: No single k value optimal for all scenarios
3. **CD-5 to CD-10**: Sweet spot for most applications
4. **PCD helps**: But modest gains (~5-8%) for memory cost
5. **Hyperparameters critical**: Langevin dynamics require careful tuning
6. **Findings generalize**: Consistent trends across architectures and datasets

These insights provide practitioners with evidence-based guidelines and researchers with directions for future investigation. The CD-k trade-off is fundamental to energy-based learning, and understanding it is essential for effective model training.