# Noise-Robust Conceptors for Physical Reservoir Computing: Adaptation to Perturbations

Conceptors are a powerful neuromorphic extension of reservoir computing that enable the selective recall and stabilization of internal dynamics. Despite their potential, conceptors have not yet been applied or adapted to physical hardware, where internal states are inevitably affected by noise and hardware imperfections. In this work, we propose a hardware-compatible adaptation of conceptors based on cross-trial correlations (CTC). By leveraging the consistency of reservoir responses across repeated trials, the method suppresses uncorrelated noise during conceptor computation. Numerical simulations under realistic conditions demonstrate that CTC-based conceptors significantly enhance robustness, preserve internal dynamics, and extend the operational performance of the reservoir compared to both standard conceptors and unconstrained reservoirs. In particular, when the noise level exceeds 50\%, reservoirs without conceptors or using standard conceptors are unable to sustain autonomous generation, whereas the proposed CTC-based conceptor enables the network to maintain predictive capability. Furthermore, under combined noise and parameter drift, the predicted output obtained with the proposed methodology maintains NRMSE values below 0.3, while previous approaches typically exceed this error level. These results provide a practical pathway for deploying conceptors in physical reservoir computing systems.

## Benchmark Task

The benchmark task consists of one-step-ahead prediction and autonomous generation of the Rössler chaotic dynamics. The goal is to evaluate the proposed noise-robust conceptors against the no-conceptor case, standard conceptors, and an averaged-state conceptor ($C_{\text{avg}}$) under noisy conditions. The $C_{\text{avg}}$ approach computes the conceptor from a new state matrix obtained by averaging the reservoir states across repeated trials before calculating its correlation matrix.

The benchmark also assesses the ability of the proposed method to improve the robustness of leaky ESNs under hardware-relevant perturbations, such as parameter drift during operation. Some experiments included in this repository additionally use an ideal conceptor computed from noise-free reservoir states as a reference. These comparisons are provided only to contextualize the performance of the different methods and are not part of the results presented in the paper.

---

## Task 1: Performance under additive noise

In this task, Gaussian noise is injected into the internal states of the ESN to emulate the intrinsic noise present in analog reservoir computing implementations. The objective is to assess whether conceptors can mitigate the impact of noise on reservoir dynamics and improve output stability.

The output weights ($W_{\text{out}}$) are trained using noisy reservoir states, reflecting realistic hardware conditions where noise is present during both training and inference. In the conceptor-free case, $W_{\text{out}}$ is trained directly on noisy states, while in the conceptor-based cases it is trained on the corresponding conceptor-projected noisy states.

The proposed CTC-based conceptor is compared with two alternative ways of computing a conceptor under noisy conditions. The standard noisy conceptor is calculated directly from the states recorded during a noisy trial. By contrast, $C_{\text{avg}}$ is computed from a new state matrix obtained by averaging the reservoir states across repeated trials. This averaging reduces uncorrelated noise before the state correlation matrix and the corresponding conceptor are calculated. The comparisons with $C_{\text{avg}}$ are included in the paper.

Some repository experiments also include an ideal conceptor computed from noise-free reservoir states. This ideal conceptor is used exclusively as an additional reference to assess how closely the different noisy conceptor estimates approach the noise-free case. It is not included in the paper and does not represent a hardware-compatible method, since noise-free internal states would not be available in a physical implementation.

Performance is first evaluated in **open-loop mode**, comparing noisy ESNs, standard noisy conceptors, averaged-state conceptors, CTC-based conceptors, and, in selected repository experiments, the ideal conceptor reference. This analysis shows how effectively each method filters noise from the reservoir states. In particular, as the noise level increases, the standard conceptor tends toward the identity matrix, effectively behaving like the no-conceptor case. The averaging used to compute $C_{\text{avg}}$ reduces part of the uncorrelated noise, while the CTC-based conceptor directly exploits the consistency between reservoir responses across repeated trials to recover a more robust representation of the underlying dynamics.

The system is subsequently evaluated in **autonomous mode**, where the effects of noise accumulate over time and have a stronger impact on long-term stability. In this regime, the CTC-based conceptor clearly outperforms both the no-conceptor and standard-conceptor cases, demonstrating that it not only filters noise from the internal states but also significantly enhances the overall performance and stability of the network. Comparisons with $C_{\text{avg}}$ provide an additional benchmark for determining whether the cross-trial formulation offers benefits beyond directly averaging the recorded states.

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



### Conceptors: basic examples

- **`Conceptors_Rossler.py`**  
  Computes and visualizes a conceptor for the internal states of a reservoir ($N=3$) driven by the Rössler chaotic time series.

- **`Conceptors_sw.py`**  
  Computes and visualizes a conceptor for the internal states of a reservoir ($N=3$) driven by a sine wave.



### Experiments under additive noise

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
  Evaluates $p$-steps-ahead prediction under noise. Reports NRMSE over multiple trials for $-25 < p < 25$ and a given level of noise.

- **`ESN_denoise_sw.py`**  
  Noise robustness analysis using a sine wave input. Compares autonomous outputs and PCA representations for:
  - no noise  
  - noise without conceptor  
  - noisy conceptor  
  - CTC-based conceptor  

- **`PCA_rossler_noise.py`**  
  Visualizes the evolution of reservoir states under increasing noise levels using PCA, comparing no conceptor, noisy conceptor, and CTC-based conceptor.



### Experiments under parameter drift


- **`ESN_drift_Rossler.py`**  
  Introduces noise and **uniform drift**, and evaluates one-step-ahead prediction performance with and without CTC-based conceptors. Example outputs are visualized.

- **`ESN_drift_Rossler_new.py`**  
  Introduces noise and **random drift**, evaluating performance under the same conditions as above.

- **`ESN_drift_heatmap_Rossler.py`**  
  Evaluates performance under **uniform drift** across a range of noise and drift levels. Generates heatmaps of NRMSE comparing no conceptor vs. CTC-based conceptor.

- **`ESN_drift_heatmap_Rossler_new.py`**  
  Same as above, but for **random drift**. Produces heatmaps showing performance across noise–drift combinations.



### Extended analysis

- **`Extended_analysis/`**  
  Contains supplementary analyses used to study the behaviour and limitations of CTC-based conceptors in greater detail. These scripts extend the main benchmark experiments and save their generated figures and data in **`Extended_analysis/plots_analysis/`**.

  - **`CTC_analysis.py`**  
    Studies how performance changes with the number of repeated reservoir-state realizations ($m$) used to compute the CTC conceptor. It evaluates PCA subspace similarity and NRMSE for different noise conditions.

  - **`aperture_analysis.py`**  
    Analyses the influence of the conceptor aperture on PCA subspace similarity and prediction NRMSE, including comparisons across different prediction horizons.

  - **`cleaning_C_fail.py`**  
    Investigates whether thresholding the singular or eigenvalue spectrum can remove noise-related components from the conceptor. It evaluates the resulting changes in PCA subspace similarity and NRMSE.

  - **`eigenvalues_analysis.py`**  
    Examines the eigenvalue spectrum of the cross-trial correlation matrix as the noise level increases, distinguishing between the components retained and removed during the construction of the CTC conceptor.

  - **`sensitivity_analysis.py`**  
    Evaluates the sensitivity of the autonomous prediction horizon to the parameters used in its calculation, including the error threshold, the required number of consecutive steps above the threshold, and the averaging-window length.

  - **`structural_analysis.py`**  
    Studies how increasing noise affects the singular-value spectrum of the conceptors and the trained readout parameters, including the magnitude and variability of $W_{\text{out}}$ and the output bias.

### Results

- **`plots/`**  
  Contains the figures generated by the different scripts, corresponding to those presented in the paper.

---

### Notes

Feel free to tune the different hyperparameters (e.g., reservoir size, spectral radius, leakage rate, noise level, or conceptor aperture) to explore new configurations and obtain additional results. The code is intended as a flexible starting point for further experimentation and extension.
