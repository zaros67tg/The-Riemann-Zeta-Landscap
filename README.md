# The-Riemann-Zeta-Landscap
# Abelian Sandpile Model
## Self-Organized Criticality — Complete Implementation

---

## Files

| File | Description |
|------|-------------|
| `sandpile_cpu.py` | Core CPU engine — NumPy vectorized, power law MLE fitter, full visualization |
| `sandpile_gpu.py` | GPU/CUDA engine — CuPy + custom CUDA kernels + Numba JIT fallback |
| `sandpile_demo.py` | Demo suite — fractals, convergence, universality, animation |

---

## Quick Start

```bash
pip install numpy matplotlib scipy

# Run full demo (generates 4 plots)
python sandpile_demo.py

# Run with 100k drops on a 300×300 grid
python sandpile_demo.py --drops 100000 --grid 300

# Just the power law analysis
python sandpile_demo.py --demo powerlaw --drops 50000

# Full CPU simulation with analysis plot
python sandpile_cpu.py
```

---

## GPU / CUDA (Optional — 10-50× faster)

```bash
# CUDA 12.x
pip install cupy-cuda12x numba

# CUDA 11.x
pip install cupy-cuda11x numba

# Run GPU simulation (1M grains)
python sandpile_gpu.py
```

If no GPU is available, `sandpile_gpu.py` automatically falls back to:
1. **Numba JIT** (parallel CPU, ~3-8× faster than NumPy)
2. **NumPy** (always available)

---

## The Physics

### Self-Organized Criticality (SOC)
The sandpile self-organizes to a **critical state** where:
- Avalanche sizes follow a **power law**: P(s) ∝ s⁻ᵅ
- No characteristic scale exists (scale-free)
- A single grain can trigger a system-spanning cascade

### Power Law Exponent
| System | α |
|--------|---|
| 2D Abelian Sandpile (theory) | ~1.0–1.5 |
| Earthquakes (Gutenberg-Richter) | ~1.0 |
| Forest fires | ~1.3 |
| Stock market crashes | ~1.5–3.0 |

### Why "Abelian"?
The final stable state is **independent of the order** in which topplings
are applied. This commutativity (Abelian property) is what makes the
model analytically tractable and guarantees the simulation is correct
even with parallel/vectorized updates.

---

## CUDA Kernel Design

```
Naive approach (WRONG): update grid in-place → data race between adjacent cells
Correct approach: 
  1. Compute topple_mask from current grid (read-only)
  2. Apply all topplings simultaneously using topple_mask (write)
  3. Repeat until no unstable cells remain
```

This synchronous update is valid because of the Abelian property:
the order of topplings does not affect the final result.

For even higher throughput, a **checkerboard (red-black)** scheme can
be used, where only cells at positions (x+y) % 2 == 0 topple in even
steps and the other color in odd steps, guaranteeing no adjacency conflicts.

---

## Fitting Method

The power law exponent α is estimated using the **maximum likelihood
estimator** (Clauset, Shalizi & Newman 2009):

```
α̂ = 1 + n [ Σᵢ ln(xᵢ / (x_min - 0.5)) ]⁻¹
```

The lower cutoff `s_min` is chosen to minimize the Kolmogorov-Smirnov
distance between the empirical and theoretical distributions.

---

## References

1. Bak, P., Tang, C., & Wiesenfeld, K. (1987). *Self-organized criticality:
   An explanation of the 1/f noise.* Physical Review Letters, 59(4), 381.

2. Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). *Power-law
   distributions in empirical data.* SIAM Review, 51(4), 661-703.

3. Dhar, D. (1990). *Self-organized critical state of sandpile automaton
   models.* Physical Review Letters, 64(14), 1613.
