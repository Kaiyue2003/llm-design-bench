# Method provenance

These runs use the configurations in raw_runs.csv. The additional methods are adaptations, not validated reproductions of published tables.

## Best Logged

- Implementation: `native_baseline`
- Source: native baseline
- Source revision: `not pinned`

## COM

- Implementation: `lightweight_adaptation`
- Source: https://github.com/rail-berkeley/design-baselines
- Source revision: `not pinned`
- explicit multi-fidelity context conditioning
- continuous simplex and box search spaces

## BDI

- Implementation: `lightweight_adaptation`
- Source: https://github.com/GGchen1997/BDI
- Source revision: `not pinned`
- finite RBF kernel replaces the official infinite-width NTK
- explicit multi-fidelity context conditioning

## Offline MLP

- Implementation: `native_baseline`
- Source: native baseline
- Source revision: `not pinned`
- explicit multi-fidelity context conditioning

## CbAS

- Implementation: `lightweight_adaptation`
- Source: https://github.com/dhbrookes/CbAS
- Source revision: `725805a2bd889084a2a9e9032d24d409fe4d7e61`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- Gaussian conditional VAE for continuous designs
- decoder density ratio uses the shared latent draw; log ratio clipped to +/-20

## MINs

- Implementation: `lightweight_adaptation`
- Source: https://github.com/rail-berkeley/design-baselines
- Source revision: `785dbcfa58107bfcc426257a1c2e69d7f71c3c27`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- weighted conditional GAN with mismatched-pair negatives
- finite target-utility search ranked by ensemble LCB and discriminator plausibility

## DDOM

- Implementation: `lightweight_adaptation`
- Source: https://github.com/siddarthk97/ddom
- Source revision: `fb3d0558cf1af153568b6fa7902c0505356821dc`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- cosine discrete VP schedule and deterministic DDIM replace the upstream continuous SDE solver
- exponential utility reweighting instead of histogram-smoothed weights

## GABO

- Implementation: `lightweight_adaptation`
- Source: https://github.com/michael-s-yao/gabo
- Source revision: `61a44c09b22645c21f06394dc04983507404eda9`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- conditional VAE latent representation with an adversarial Wasserstein source critic
- finite-grid dual coefficient; exact RBF GP with median lengthscale and sequential analytic EI instead of BoTorch qEI

## GTG

- Implementation: `lightweight_adaptation`
- Source: https://github.com/dbsxodud-11/GTG
- Source revision: `448f20635b3e484d446ffe091d1861136068fc8c`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- within-fidelity nearest-neighbor improving trajectories
- flattened trajectory MLP diffusion instead of temporal U-Net; mean-return CFG and anchored start

## RGD

- Implementation: `lightweight_adaptation`
- Source: https://github.com/GGchen1997/RGD
- Source revision: `c8ab225e09999534651e27be634921af5f2ef002`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- cosine VP diffusion with a compact Gaussian ensemble proxy
- midpoint probability-flow likelihood with a seeded Hutchinson trace and terminal t=.98
- clipped reverse-KL proxy refinement; normalized gradients on denoised samples

## BONET

- Implementation: `lightweight_adaptation`
- Source: https://github.com/siddarthk97/bonet
- Source revision: `14b506a695c168dea6ddaae8867cef41859fe716`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- within-fidelity bootstrap-sorted trajectories and Gaussian causal transformer
- strictly offline rollout updates regret using a trained proxy instead of online oracle evaluations

## DEMO

- Implementation: `lightweight_adaptation`
- Source: https://github.com/mila-iqia/Design-Editing-for-Offline-MBO
- Source revision: `3f02bec3a64e19b0dd36884b258ac014fd1bf0ec`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- surrogate-gradient pseudo-targets and target-distribution diffusion fine-tuning
- partial cosine noising and DDIM editing replace the upstream VP-SDE/Heun solver

## ROOT

- Implementation: `lightweight_adaptation`
- Source: https://github.com/cuong-dm/ROOT
- Source revision: `d23f14fe30d53f1fc4423ce006056672d0353906`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- linear Brownian bridge with displacement objective and analytic stochastic reverse transitions
- within-fidelity quantile pairing and compact time-conditioned MLP

## SPADE

- Implementation: `lightweight_adaptation`
- Source: https://github.com/HarryYoung2018/spade
- Source revision: `586151bbb56e246f93ca97ce33f79887a13161bd`
- independent continuous PyTorch implementation; paper-result parity not established
- train-only standardized box/ALR features and explicit fidelity conditioning
- scalar conditional diffusion with moment/rank and support-proximity training losses
- cosine schedule, compact MLP, exact torch kNN, and seeded evolutionary LCB search
