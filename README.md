# Noise-Robust Conceptors for Physical Reservoir Computing

Conceptors are a powerful neuromorphic extension of reservoir computing that enable the selective recall and stabilization of internal dynamics. Despite their potential, conceptors have not yet been applied or adapted to physical hardware, where internal states are inevitably affected by noise and hardware imperfections. In this work, we propose a hardware-compatible adaptation of conceptors based on cross-trial correlations (CTC). By leveraging the consistency of reservoir responses across repeated trials, the method suppresses uncorrelated noise during conceptor computation. Numerical simulations under realistic conditions demonstrate that CTC-based conceptors significantly enhance robustness, preserve internal dynamics, and extend the operational performance of the reservoir compared to both standard conceptors and unconstrained reservoirs. In particular, when the noise level exceeds 50\%, reservoirs without conceptors or using standard conceptors are unable to sustain autonomous generation, whereas the proposed CTC-based conceptor enables the network to maintain predictive capability. Furthermore, under combined noise and parameter drift, the predicted output obtained with the proposed methodology maintains NRMSE values below 0.3, while previous approaches typically exceed this error level. These results provide a practical pathway for deploying conceptors in physical reservoir computing systems.

## Benchmark Task

The benchmark task consists of one step predicting one-step-ahead and autonomous generation of the Rössler chaotic dynamics. The goal is to evaluate the proposed noise-robust conceptors against both the no-conceptor case and standard conceptors under noisy scenarios, as well as to assess their ability to improve the robustness of leaky ESNs under hardware-relevant conditions as parameter drift during operation.

---

## Task 1: Performance under additive noise

In this task, Gaussian noise is injected into the internal states of the ESN to emulate the intrinsic noise present in analog reservoir computing implementations. The objective is to assess whether conceptors can mitigate the impact of noise on reservoir dynamics and improve output stability.

The output weights ($W_{\text{out}}$) are trained using noisy reservoir states, reflecting realistic hardware conditions where noise is present during both training and inference. In the conceptor-free case, $W_{\text{out}}$ is trained directly on noisy states, while in the conceptor-based cases they are trained on the corresponding conceptor-projected noisy states.

Performance is first evaluated in **open-loop mode**, comparing noisy ESNs, standard conceptors, CTC-based conceptors, and a noise-free reference. This analysis shows how each method filters noise in the reservoir states. In particular, as the noise level increases, the standard conceptor tends toward the identity matrix, effectively behaving like no conceptor at all. In contrast, the CTC-based conceptor continues to effectively filter noise, producing cleaner internal representations even at high noise levels.

Subsequently, the system is evaluated in **autonomous mode**, where noise effects accumulate over time and have a stronger impact on long-term stability. In this regime, the CTC-based conceptor clearly outperforms both the no-conceptor and standard conceptor cases, demonstrating that it not only denoises the internal states but also significantly enhances the overall performance and stability of the network.

---

## Task 2: Robustness under parameter drift

This task evaluates the robustness of the ESN under **hardware-like parameter drift** occurring after training. The drift is introduced in the bias vector during operation, modifying neuron activations and degrading performance. The output weights are kept the same as in the previous task, meaning they are trained under noisy conditions without accounting for the subsequent drift.

Two types of drift are considered: **uniform drift**, where all neurons are affected equally, and **random drift**, where each neuron experiences independent perturbations. After an initial period of normal operation, the drift is introduced and the system is evaluated in a one-step-ahead prediction task of the Rössler system.

Results show that parameter drift significantly degrades the performance of standard ESNs. However, the CTC-based conceptor consistently improves robustness, maintaining outputs closer to the target dynamics under both drift types. This demonstrates that the proposed approach enhances stability against hardware-induced degradation effects.


---

## Repository Structure

### Core utilities

- **`utils/`**
  - **`rrnn_utils.py`**  
    Functions for training the reservoir, computing internal states, and calculating conceptors.

  - **`utils.py`**  
    Utility functions for visualization and result presentation.

---

### Conceptors: basic examples

- **`Conceptors_Rossler.py`**  
  Computes and visualizes a conceptor for the internal states of a reservoir ($N=3$) driven by the Rössler chaotic time series.

- **`Conceptors_sw.py`**  
  Computes and visualizes a conceptor for the internal states of a reservoir ($N=3$) driven by a sine wave.

---

### Experiments under additive noise

These scripts analyze the effect of noise on the reservoir dynamics and evaluate the performance of standard conceptors, CTC-based conceptors, and the no-conceptor case.

- **`ESN_denoise_atractor_Rossler_aut.py`**  
  Studies the autonomous reconstruction of the Rössler attractor under noise. Compares performance with no conceptor, noisy conceptor, and CTC-based conceptor.

- **`ESN_denoise_loop_Rossler_aut.py`**  
  Evaluates autonomous performance over multiple trials (100 runs with different seeds). Generates:
  - Boxplots of maximum cross-correlation of the frequency spectrum with the noise-free reference  
  - Boxplots of prediction horizon  
  - Example outputs and spectra for selected noise levels  

- **`ESN_denoise_loop_Rossler_ol.py`**  
  Evaluates one-step-ahead (open-loop) performance under noise. Computes the similarity of internal states using PCA and compares against the noise-free case across multiple trials.

- **`ESN_denoise_pd_Rossler_ol.py`**  
  Evaluates $p$-steps-ahead prediction under noise. Reports NRMSE over multiple trials for a range of prediction horizons.

- **`ESN_denoise_sw.py`**  
  Noise robustness analysis using a sine wave input. Compares autonomous outputs and PCA representations for:
  - no noise  
  - noise without conceptor  
  - noisy conceptor  
  - CTC-based conceptor  

- **`PCA_rossler_noise.py`**  
  Visualizes the evolution of reservoir states under increasing noise levels using PCA, comparing no conceptor, noisy conceptor, and CTC-based conceptor.

---

### Experiments under parameter drift

These scripts evaluate robustness against hardware-like degradation by introducing drift in the reservoir parameters, together with noise.

- **`ESN_drift_Rossler.py`**  
  Introduces noise and **uniform drift**, and evaluates one-step-ahead prediction performance with and without CTC-based conceptors. Example outputs are visualized.

- **`ESN_drift_Rossler_new.py`**  
  Introduces noise and **random drift**, evaluating performance under the same conditions as above.

- **`ESN_drift_heatmap_Rossler.py`**  
  Evaluates performance under **uniform drift** across a range of noise and drift levels. Generates heatmaps of NRMSE comparing no conceptor vs. CTC-based conceptor.

- **`ESN_drift_heatmap_Rossler_new.py`**  
  Same as above, but for **random drift**. Produces heatmaps showing performance across noise–drift combinations.

---

### Results

- **`plots/`**  
  Contains the figures generated by the different scripts, corresponding to those presented in the paper.

---

### Notes

Feel free to tune the different hyperparameters (e.g., reservoir size, spectral radius, leakage rate, noise level, or conceptor aperture) to explore new configurations and obtain additional results. The code is intended as a flexible starting point for further experimentation and extension.
