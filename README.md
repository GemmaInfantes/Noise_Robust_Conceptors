# Noise-Robust Conceptors for Physical Reservoir Computing

Conceptors are a powerful neuromorphic extension of reservoir computing that enable the selective recall and stabilization of internal dynamics. Despite their potential, conceptors have not yet been applied or adapted to physical hardware, where internal states are inevitably affected by noise and hardware imperfections. In this work, we propose a hardware-compatible adaptation of conceptors based on cross-trial correlations (CTC). By leveraging the consistency of reservoir responses across repeated trials, the method suppresses uncorrelated noise during conceptor computation. Numerical simulations under realistic conditions demonstrate that CTC-based conceptors significantly enhance robustness, preserve internal dynamics, and extend the operational performance of the reservoir compared to both standard conceptors and unconstrained reservoirs. In particular, when the noise level exceeds 50\%, reservoirs without conceptors or using standard conceptors are unable to sustain autonomous generation, whereas the proposed CTC-based conceptor enables the network to maintain predictive capability. Furthermore, under combined noise and parameter drift, the predicted output obtained with the proposed methodology maintains NRMSE values below 0.3, while previous approaches typically exceed this error level. These results provide a practical pathway for deploying conceptors in physical reservoir computing systems.

---

## Benchmark Task

The benchmark task consists of predicting and generating the Rössler chaotic dynamics in autonomous mode. The goal is to evaluate the proposed noise-robust conceptors against standard conceptors under noisy scenarios, as well as to assess their ability to improve the robustness of leaky ESNs under hardware-relevant conditions. The CTC-based conceptors are evaluated under two scenarios: additive noise and parameter drift.

---

## Task 1: Performance under additive noise

In this task, Gaussian noise is injected into the internal states of the ESN to emulate the intrinsic noise present in analog reservoir computing implementations. The objective is to assess whether conceptors can mitigate the impact of noise on reservoir dynamics and improve output stability.

The output weights ($W_{\text{out}}$) are trained using noisy reservoir states, reflecting realistic hardware conditions where noise is present during both training and inference. In the noise-free case, $W_{\text{out}}$ is trained on clean states, while in the conceptor-based case it is trained on conceptor-projected noisy states.

Performance is evaluated first in **open-loop mode**, comparing noisy ESNs, standard conceptors, CTC-based conceptors, and the noise-free reference. This allows assessing how effectively each method filters noise in the reservoir states. Subsequently, the system is evaluated in **autonomous mode**, where noise effects accumulate over time and have a stronger impact on long-term stability.

---

## Task 2: Rubustess under parameter drift

This task evaluates the robustness of the ESN under **hardware-like parameter drift** occurring after training. The drift is introduced in the bias vector during operation, modifying neuron activations and degrading performance.

Two types of drift are considered: **uniform drift**, where all neurons are affected equally, and **random drift**, where each neuron experiences independent perturbations. After an initial period of normal operation, the drift is activated and the system is tested in a one-step-ahead prediction task of the Rössler system.

Results show that parameter drift significantly degrades the performance of standard ESNs. However, the CTC-based conceptor consistently improves robustness, maintaining outputs closer to the target dynamics under both drift types, even when noise is present in the reservoir states. This demonstrates that the proposed approach enhances stability against hardware-induced degradation effects.



---
## Repository Structure

- **`utils/`**
  - **`rrnn_utils.py`**  
    Contains functions for training the reservoir, computing internal states, and calculating conceptors.

  - **`utils.py`**  
    Provides functions for visualization and result presentation.

- **`Conceptors_sw.py`**  
  Computes and visualizes a conceptor for the internal states of a reservoir ($N=3$) driven by a sine wave.

- **`ESN_denoise_loop_aut.py`**  
  Introduces noise into the internal states and studies the performance of the reservoir with and without conceptors. A boxplot of the maximum corss correlation over 50 trials (using different random seeds) is generated for increasing noise levels, together with example output signals for a fixed noise level.

- **`network_degradation.py`**  
  Implements manual network degradation and visualizes the corresponding outputs for the cases with and without conceptors.

- **`network_degradation_loop.py`**  
  Performs a quantitative evaluation of performance, analogous to the noise boxplot analysis, under progressive network degradation by randomly removing $K$ neurons across multiple trials. The performance of the reservoir with and without conceptors is systematically compared.


- **`plots/`**  
  Stores the final figures generated by the different `.py` scripts, corresponding to those shown in the paper.

---

### Notes

Feel free to tune the different hyperparameters (e.g., reservoir size, spectral radius, leakage rate, noise level, or conceptor aperture) to explore new configurations and obtain additional results. The code is intended as a flexible starting point for further experimentation and extension.
