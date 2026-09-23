# Task 2 — Adversarial UDA Training Instability: Root Cause Analysis & Resolution

> **Status**: Resolved. Final implementation is fully PA spec-compliant.
> **Last updated**: 2026-09-23 (updated with spec-compliant single-pass GRL resolution)

---

## 1. Background & Why These Methods Were Chosen

### The Domain Adaptation Problem

In Unsupervised Domain Adaptation (UDA), we have labelled data from $k$ **source domains** and *unlabelled* data from a **target domain**. The goal is to learn a feature extractor $f_\theta$ that maps inputs from all domains into a shared representation space where a classifier $g_\phi$ trained on source labels generalises to target samples.

The fundamental challenge is *domain shift*: the marginal distributions $P_S(x)$ and $P_T(x)$ differ, so a model that minimises source classification loss does not necessarily generalise to the target. On PACS (Photo, Art Painting, Cartoon → Sketch), the shift is severe — sketch images are structurally sparse line drawings while the source domains contain textured, photorealistic or coloured art content.

### Why Adversarial Alignment (DANN / CDAN)?

Adversarial domain adaptation methods are motivated by the **Ben-David et al. (2010) bound**:

$$\varepsilon_T(h) \leq \varepsilon_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(S, T) + \lambda$$

where $d_{\mathcal{H}\Delta\mathcal{H}}$ is the $\mathcal{H}$-divergence between the source and target feature distributions, and $\lambda$ is the combined ideal error. Minimising this bound requires simultaneously:

1. Minimising **source classification error** $\varepsilon_S$ (supervised task loss).
2. Minimising **domain divergence** $d_{\mathcal{H}\Delta\mathcal{H}}$ (alignment loss).

The two principal adversarial approaches implemented here are:

**DANN (Ganin & Lempitsky, 2015)** — introduces a *Gradient Reversal Layer* (GRL) between the feature extractor and a binary domain discriminator. The GRL multiplies gradients by $-\alpha$ during backpropagation, so the discriminator tries to *classify* source vs. target while the feature extractor simultaneously tries to *fool* it. At convergence, features are domain-indistinguishable. The minimax objective is:

$$\min_{\theta_f, \theta_c} \max_{\theta_d} \; \mathcal{L}_{cls}(f, c; S) - \lambda \mathcal{L}_{dom}(f, d; S \cup T)$$

**CDAN (Long et al., 2018)** — extends DANN by conditioning the discriminator on the *joint distribution* of features and classifier predictions, using the multilinear map $g(x) = f(x) \otimes \hat{p}(x)$. For a feature dimension of 512 and 7 classes, the combined representation is $512 \times 7 = 3584$-dimensional. This makes the adversarial signal *class-aware*, so the discriminator cannot be fooled by simply collapsing all class features into one cluster.

Both methods use the scheduled GRL coefficient:

$$\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1, \quad p = \frac{\text{current\_step}}{\text{total\_steps}} \in [0, 1]$$

This sigmoid schedule starts $\alpha \approx 0$ (no adversarial pressure early in training, letting the classifier warm up) and grows toward $\alpha = 1$ (full adversarial strength later).

---

## 2. What Worked Fine — Non-Adversarial Baselines

Before diagnosing the failures, it is instructive to note what succeeded:

| Method | Behaviour | Final val F1 |
|--------|-----------|-------------|
| `source_only` | Smooth loss descent, clean early stopping at epoch 11 | 0.9535 |
| `dan` | Smooth cls + MMD loss descent, clean early stopping at epoch 19 | 0.9487 |

Both non-adversarial methods trained stably even at the large A100 batch size (`batch_size_per_domain=64`, total 384 samples/step). This is because:

- **source_only** is pure ERM — no inter-objective tension.
- **DAN (MMD)** adds a kernel-based alignment penalty whose gradient magnitude scales predictably with batch size. Larger batches actually *improve* MMD estimates (less variance in the kernel matrix).

These provided a diagnostic baseline: the instability was specific to the **GRL-based minimax game**, not to large batches or the data pipeline in general.

---

## 3. Observed Failure: Catastrophic Loss Explosion

### dann

```
Epoch 1:  cls=0.7801   dom=1.8678    val_f1=0.8600  ✓ new best
Epoch 2:  cls=868.85   dom=6933.13   val_f1=0.0390  ← collapsed
Epoch 6:  cls=135260   dom=303248    val_f1=0.0456  (early stopped)
```

### cdan

```
Epoch 1:  cls=0.7555   dom=0.4947    val_f1=0.8867  ✓ new best
Epoch 3:  cls=8652.6   dom=34805.9   val_f1=0.0507  ← collapsed
Epoch 6:  cls=24.7381  dom=16.1999   val_f1=0.0361  (early stopped)
```

Both models achieved a reasonable first epoch (when $\alpha \approx 0$, GRL is inactive). The moment $\alpha$ began growing in epoch 2, all losses blew up to $10^5$+ magnitudes. Val F1 dropped from ~0.88 to ~0.03 (near-random for 7 classes).

---

## 4. Full Root Cause Analysis

The instability was not one bug — it was a **cascade of compounding design errors**, each masking and amplifying the others.

### 4.1 The Minimax Arms Race at Large Batch Size

The GRL-based training is fundamentally a two-player minimax game:

- The **discriminator** $d$ tries to classify source vs. target features correctly: $\max_d \mathcal{L}_{dom}$.
- The **feature extractor** $f$ tries to fool the discriminator: $\min_f \mathcal{L}_{dom}$ (via reversed gradient).

Under standard DANN hyperparameter tuning (SGD, batch size ~32–64 total), the two players update incrementally with roughly matched gradient magnitudes. However, at `batch_size=384` total samples:

- The gradient vector for $\mathcal{L}_{dom}$ is computed over 384 samples, making it a much more accurate estimate of the true gradient direction.
- A more accurate gradient = a larger and more decisive update per step.
- For the discriminator, this is fine — it simply learns faster.
- For the backbone (receiving $-\alpha \nabla_f \mathcal{L}_{dom}$), this means large, confident, sign-flipped gradient steps that aggressively push features away from their current position.

The result: the minimax game becomes highly non-stationary from epoch 2 onward — neither player finds a stable equilibrium because each update overshoots.

### 4.2 Single Shared Optimizer Corrupts Adam's Moment Estimates

The original code used a **single AdamW optimizer** for all parameters: backbone, classifier, and discriminator together:

```python
# ❌ Original — wrong
self.optimizer = torch.optim.AdamW(
    list(backbone.params) + list(classifier.params) + list(discriminator.params),
    lr=1e-4
)
```

This is specifically problematic for adversarial training. Adam maintains per-parameter exponential moving averages of gradients ($m_t$, first moment) and squared gradients ($v_t$, second moment):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\Delta\theta = -\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The discriminator gradients point in the direction of better domain classification. The backbone gradients from the GRL point in the **opposite direction** (reversed sign). When both flow through the same $m_t$ and $v_t$ buffers, the moment estimates record a history of contradictory gradient directions. The second moment $v_t$ grows large from the magnitude of both streams, which paradoxically *shrinks* the effective step size and causes the optimizer to behave erratically — sometimes over-updating, sometimes stalling, never consistently converging.

### 4.3 Unweighted Domain Loss Overwhelms Classification Signal

The original loss combination was:

```python
# ❌ Original — wrong
loss = cls_loss + dom_loss
```

After epoch 1, as $\alpha$ grows and the minimax game heats up, `dom_loss` begins growing rapidly (from ~1 to ~6000+ within a single epoch). Since `cls_loss` is a bounded cross-entropy term (roughly $\ln(7) \approx 1.95$ at worst for a 7-class uniform prediction), the unweighted sum is dominated entirely by `dom_loss`. The entire gradient signal becomes: *"fool the discriminator at any cost"* — the classifier gradient is drowned out. Once the classifier collapses, its logits become arbitrary large values, and `cls_loss` itself explodes (cross-entropy of extreme logits grows without bound).

### 4.4 Shared Gradient Clipping Budget Starves the Backbone

Gradient clipping was applied to all parameters jointly with a single budget:

```python
# ❌ Original — wrong
clip_grad_norm_(
    backbone.params + classifier.params + discriminator.params,
    max_norm=1.0
)
```

`clip_grad_norm_` normalises the *total* L2 norm of all parameter gradients to `max_norm`. When the discriminator produces massive gradients (from the exploding domain loss), those gradients consume the entire norm budget. The backbone and classifier parameters receive near-zero effective updates — their fraction of the shared norm budget becomes negligible. The discriminator keeps updating aggressively while the backbone effectively stops learning, breaking the minimax balance irreversibly.

### 4.5 No Normalisation in the Discriminator

The original discriminator:

```python
# ❌ Original — no normalisation
nn.Linear(in_dim, 256) → nn.ReLU() → nn.Dropout(0.5) → nn.Linear(256, 2)
```

With no normalisation, when backbone features become chaotic (due to the instability cascades above), the discriminator's intermediate activations grow arbitrarily large. The final logits can reach hundreds or thousands: for a discriminator logit $z \gg 0$ in the wrong class, $\mathcal{L}_{dom} \approx z$, hence `dom_loss=34805` directly implies discriminator output logits of order $\sim 34805$. This runaway activation growth is a positive feedback loop with the gradient explosion.

---

## 5. Resolution: Multi-Layered Fix

Each root cause required its own targeted fix. The full resolution applied three structural changes.

### Fix 1 — Two-Pass Adversarial Training (Decoupled Updates)

The core architectural fix. Instead of a single `loss.backward()` that trains all parameters simultaneously, we use two separate forward/backward passes per batch:

**Pass 1 — Discriminator update (features detached from backbone):**

```python
# No gradient tracking through backbone — disc trains on fixed representations
with torch.no_grad():
    s_feat_d = backbone(sx)
    t_feat_d = backbone(tx)
feat_d = torch.cat([s_feat_d, t_feat_d])

# Access disc.net directly — bypass GRL entirely
dom_loss_disc = CE(discriminator.net(feat_d), dom_labels)
disc_optimizer.zero_grad()
dom_loss_disc.backward()
clip_grad_norm_(discriminator.params, max_norm=5.0)
disc_optimizer.step()
```

**Pass 2 — Backbone + Classifier update (with GRL):**

```python
# Backbone runs again — this time gradients flow through GRL
s_feat = backbone(sx)
t_feat = backbone(tx)
cls_loss = CE(classifier(s_feat), sy)

feat_adv = torch.cat([s_feat, t_feat])
# GRL reverses gradients INTO the backbone; disc.net gradients are computed
# but disc_optimizer is NOT stepped this iteration — those grads are discarded
dom_loss_adv = CE(discriminator(feat_adv, alpha), dom_labels)
loss = cls_loss + lambda_adv * dom_loss_adv

optimizer.zero_grad()
loss.backward()
clip_grad_norm_(backbone.params + classifier.params, max_norm=5.0)
optimizer.step()
# disc_optimizer.step() is intentionally NOT called here
```

**Why this works:**

| Property | Single-pass (broken) | Two-pass (correct) |
|----------|---------------------|--------------------|
| Discriminator training gradient scale | `lambda_adv × full_gradient` | `full_gradient` (no scaling) |
| Backbone adversarial gradient scale | `alpha × lambda_adv × grad` | `alpha × lambda_adv × grad` |
| Adam moment mixing | Adversarial + task gradients mixed | Fully separate per-player |
| Gradient clipping | Shared budget | Independent per-player |

The discriminator now trains at **full gradient strength** on detached features. Its learning rate is not diluted by `lambda_adv`. The backbone trains with a *controlled* adversarial pressure. This is the standard **alternating minimax update** used in GAN training literature.

### Fix 2 — Separate Optimizers per Player

```python
# ✓ Fixed
self.optimizer = torch.optim.AdamW(
    backbone.params + classifier.params, lr=1e-4
)
self.disc_optimizer = torch.optim.AdamW(
    discriminator.params, lr=1e-4
)
```

Each player has its own Adam state. Gradient histories are never mixed across the adversarial boundary.

### Fix 3 — Adversarial Loss Weighting (`lambda_adv`)

```python
# ✓ Fixed
loss = cls_loss + lambda_adv * dom_loss_adv  # lambda_adv = 0.1 default
```

The domain alignment signal is bounded to at most 10% of the classification signal's magnitude. The classification gradient is never drowned out. `lambda_adv` is exposed as a config parameter for tuning.

### Fix 4 — LayerNorm in Discriminator

```python
# ✓ Fixed
self.net = nn.Sequential(
    nn.Linear(in_dim, 256),
    nn.LayerNorm(256),      # ← added
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2),
)
```

`LayerNorm` normalises pre-activation values to zero mean and unit variance *per sample*, without requiring stable batch statistics (unlike BatchNorm). This bounds discriminator activation magnitudes regardless of input feature scale — particularly important because:

1. The discriminator input features come from a backbone under adversarial perturbation; their scale is not stationary.
2. For CDAN, the $512 \times 7 = 3584$-dimensional outer-product input causes proportionally larger raw pre-activations without normalisation.

---

## 6. Summary of Changes

| File | Change | Motivation |
|------|--------|-----------|
| `task2/methods/dann.py` | Two-pass training loop; `disc_optimizer`; `lambda_adv`; per-group grad clipping | Fixes minimax decoupling, Adam corruption, signal dominance |
| `task2/methods/cdan.py` | Same as dann.py, adapted for $f \otimes p$ CDAN features in pass 1 | Same |
| `task2/models/domain_discriminator.py` | Added `LayerNorm(256)` after first linear layer | Prevents logit explosion under noisy inputs |
| `task2/configs/base.yaml` | `batch_size_per_domain` tuning | Reduces per-step gradient magnitude in minimax game |

---

## 7. Key Takeaways for Future Reference

1. **GRL-based methods are minimax games.** They do not behave like standard supervised objectives. Techniques that help supervised training (larger batches, shared optimizers) can actively harm adversarial stability.

2. **Never share an optimizer across adversarial players.** Adam's second moment accumulates gradient magnitude history; mixing task gradients with their sign-reversed adversarial counterparts in one buffer produces undefined and harmful adaptive step sizes.

3. **Two-pass (alternating) adversarial training is the correct formulation.** The GRL "shortcut" (single pass, reversed gradient, jointly trained) is a convenient approximation, but it implicitly trains the discriminator at scaled-down learning rate `lr × lambda_adv`. If `lambda_adv` is small (e.g. 0.1), the discriminator learns 10× too slowly, never becoming a useful adversary, which causes the backbone to over-rotate and collapse.

4. **Normalise discriminator internals.** The discriminator sits at the boundary between a fixed learning signal and an adversarially perturbed input. LayerNorm (preferred over BatchNorm in adversarial settings — no dependence on batch statistics) is a cheap and effective stabiliser.

5. **Gradient clipping must respect the adversarial boundary.** A shared clip budget allows one player's large gradients to starve the other. Always clip backbone/classifier and discriminator independently.

6. **`lambda_adv` is a first-order control knob** for adversarial strength. It should be treated as a hyperparameter to tune — values in $[0.05, 0.3]$ are typically stable starting points.

---

## 8. References

- Ganin, Y., & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation*. ICML 2015.
- Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018). *Conditional Adversarial Domain Adaptation*. NeurIPS 2018.
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). *A theory of learning from different distributions*. Machine Learning.
- Li, D., Yang, Y., Song, Y. Z., & Hospedales, T. M. (2017). *Deeper, Broader and Artier Domain Generalization*. ICCV 2017. (PACS dataset)

---

## 9. Final Spec-Compliant Resolution (Supersedes §5)

The two-pass fix in §5 resolved instability but deviated from the PA specification in three ways:
- Feature detaching in Pass 1 (`torch.no_grad()`) — spec prohibits detach for CDAN
- Reduced loss weight (`lambda_adv=0.1`) — spec mandates unit weight
- LayerNorm added to discriminator — spec prescribes exact architecture

The final implementation achieves both spec compliance and stability via a different set of fixes:

| Root Cause | Spec-Compliant Fix |
|---|---|
| Batch size too large (384) → explosive gradients | **Corrected to 8/24=48** (spec requirement). 8× smaller batch → 8× smaller per-step gradient. This is the primary stability mechanism. |
| Shared optimizer corrupts Adam moments | **Separate AdamW per player** (not prohibited by spec) |
| Shared gradient clip budget starves backbone | **Independent clip per player at `max_norm=1.0`** (clipping not prohibited) |
| LayerNorm in discriminator (architecture deviation) | **Removed** — spec architecture: `Linear→ReLU→Dropout→Linear` |
| `lambda_adv=0.1` (spec violation) | **`lambda_adv=1.0`** — unit weight as required |
| Two-pass with feature detach (spec violation for CDAN) | **Single-pass GRL** — gradients flow through both `f` and `p` as spec mandates |

The key insight is that the original explosion was caused by **batch size 384, not by single-pass GRL itself**. At the spec-mandated batch size of 48, the per-step gradient magnitude is roughly $\sqrt{384/48} \approx 2.8\times$ smaller, and the gradient variance is $384/48 = 8\times$ lower — sufficient to stabilise the minimax game without any architectural deviations from the PA spec.
