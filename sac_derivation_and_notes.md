# Soft Actor-Critic (SAC): derivation, NumPy implementation, and philosophy note

## 1. Problem setup and the maximum-entropy objective

Let an MDP be \((\mathcal S,\mathcal A,p,r,\gamma)\), where \(s_t\) is a state, \(a_t\) an action, \(r_t=r(s_t,a_t)\), and \(\gamma\in[0,1)\). A stochastic policy \(\pi_\theta(a\mid s)\) maps a state to a distribution over actions.

Ordinary RL maximizes expected discounted reward:

\[
J_{\mathrm{reward}}(\pi)=\mathbb E_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
\]

SAC instead maximizes reward **and** policy entropy:

\[
J_{\mathrm{SAC}}(\pi)=\mathbb E_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^t\left(r_t+\alpha\,\mathcal H(\pi(\cdot\mid s_t))\right)\right],
\]

where

\[
\mathcal H(\pi(\cdot\mid s))=-\mathbb E_{a\sim\pi(\cdot\mid s)}[\log\pi(a\mid s)]
\]

and \(\alpha\geq0\) is the temperature. Equivalently, each sampled step contributes \(r_t-\alpha\log\pi(a_t\mid s_t)\). Thus SAC rewards actions that produce return while also discouraging the policy from becoming prematurely deterministic. Entropy is a control regularizer, not an independent reward for arbitrary noise.

## 2. Soft Bellman equations

For a fixed policy, define the entropy-regularized action value

\[
Q^\pi(s,a)=\mathbb E\left[r(s,a)+\gamma V^\pi(s')\right],\quad s'\sim p(\cdot\mid s,a),
\]

with soft state value

\[
V^\pi(s)=\mathbb E_{a\sim\pi}\left[Q^\pi(s,a)-\alpha\log\pi(a\mid s)\right].
\]

Substituting the value definition gives the soft Bellman expectation equation:

\[
Q^\pi(s,a)=\mathbb E_{s',a'\sim p,\pi}\left[r(s,a)+\gamma\left(Q^\pi(s',a')-\alpha\log\pi(a'\mid s')\right)\right].
\]

In an idealized continuous-action setting, the entropy-regularized optimal policy has the Boltzmann form

\[
\pi^*(a\mid s)\propto\exp\left(\frac{Q^*(s,a)}{\alpha}\right),
\]

and its soft value is a log-partition function:

\[
V^*(s)=\alpha\log\int_{\mathcal A}\exp\left(\frac{Q^*(s,a)}{\alpha}\right)\,da.
\]

As \(\alpha\to0\), the policy approaches greedy maximization of \(Q\); larger \(\alpha\) spreads probability over more actions. In practice, SAC represents the policy with a neural network and learns the critics by bootstrapping.

## 3. Critic targets and losses

SAC uses two critics \(Q_{\phi_1},Q_{\phi_2}\) and slowly moving target critics \(Q_{\bar\phi_1},Q_{\bar\phi_2}\). For replay transition \((s,a,r,s',d)\), sample \(a'\sim\pi_\theta(\cdot\mid s')\) and construct

\[
y=r+\gamma(1-d)\left[\min_{j\in\{1,2\}}Q_{\bar\phi_j}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right].
\]

The minimum of the two target critics reduces positive value bias from function approximation. Each critic minimizes a squared Bellman residual:

\[
\mathcal L_Q(\phi_i)=\frac12\,\mathbb E_{(s,a,r,s',d)\sim\mathcal D}\left[\left(Q_{\phi_i}(s,a)-\operatorname{stopgrad}(y)\right)^2\right].
\]

The target is treated as constant while differentiating the critic loss. True terminal transitions use \(d=1\); a time-limit truncation is not marked terminal in this implementation, so value bootstrapping continues through it.

Target critics use Polyak averaging after updates:

\[
\bar\phi_i\leftarrow(1-\tau)\bar\phi_i+\tau\phi_i,
\]

with a small \(\tau\) (here \(0.005\)) to make targets move smoothly.

## 4. Squashed-Gaussian policy for bounded continuous actions

A Gaussian sampled directly can leave the action bounds. SAC therefore samples a latent Gaussian and squashes it through \(\tanh\). The actor outputs \(\mu_\theta(s)\) and \(\ell_\theta(s)=\log\sigma_\theta(s)\), with \(\ell\) clipped to \([-5,2]\). Reparameterize the sample as

\[
\epsilon\sim\mathcal N(0,I),\qquad u=\mu_\theta(s)+\exp(\ell_\theta(s))\odot\epsilon,
\]

\[
h=\tanh(u),\qquad a=b+c\odot h,
\]

where \(b=(a_{\max}+a_{\min})/2\) and \(c=(a_{\max}-a_{\min})/2\). This is differentiable with respect to actor parameters for a fixed noise draw \(\epsilon\).

Because \(a\) is a transformed random variable, its log-density must include the change-of-variables Jacobian:

\[
\begin{aligned}
\log\pi_\theta(a\mid s)=\sum_j\big[&-\tfrac12\epsilon_j^2-\ell_j-\tfrac12\log(2\pi)\\
&-\log(1-h_j^2+\delta)-\log c_j\big],
\end{aligned}
\]

where \(\delta\) is a small numerical stabilizer. Omitting the \(\tanh\) correction would compute the density of the latent Gaussian rather than the executed action; omitting \(\log c_j\) would miss action-range scaling.

## 5. Actor objective and pathwise derivatives

The actor minimizes

\[
\mathcal L_\pi(\theta)=\mathbb E_{s\sim\mathcal D,\epsilon\sim\mathcal N}\left[\alpha\log\pi_\theta(a_\theta(s,\epsilon)\mid s)-\min_i Q_{\phi_i}(s,a_\theta(s,\epsilon))\right].
\]

The entropy term favours policies with higher entropy; the Q term favours actions with higher estimated return. The reparameterization trick lets gradients pass through the sampled action, avoiding a score-function estimator in this actor update.

For each action dimension, let \(q_a=\nabla_a\min_i Q_{\phi_i}(s,a)\), \(\sigma=e^\ell\), and

\[
g=\frac{2h(1-h^2)}{1-h^2+\delta}.
\]

This is the derivative of the tanh-Jacobian correction with respect to the latent \(u\). The pathwise derivatives used by the implementation are

\[
\boxed{\nabla_\mu\mathcal L_\pi=\alpha g-q_a\odot c\odot(1-h^2)}
\]

and

\[
\boxed{\nabla_\ell\mathcal L_\pi=\alpha(-1+g\odot\sigma\odot\epsilon)-q_a\odot c\odot(1-h^2)\odot\sigma\odot\epsilon.}
\]

For the first term in the second expression, the \(-1\) comes from the Gaussian log-density \(-\ell\); the \(g\sigma\epsilon\) term comes from how the transformed sample changes with \(\ell\). The critic derivative is propagated through \(a=b+c\tanh(u)\). At the minimum of the twin critics, the gradient follows whichever critic has the smaller value for that sample. Batch gradients are averaged, and log-standard-deviation gradients are zeroed where hard clipping is active.

## 6. Automatic temperature tuning

A fixed \(\alpha\) sets one entropy/reward trade-off for every stage of learning. SAC can instead adapt it toward a chosen target entropy \(\mathcal H_*\). Since

\[
\mathcal H_\pi=-\mathbb E[\log\pi(a\mid s)],
\]

a common log-temperature surrogate is

\[
\mathcal L_\alpha(\beta)=-\mathbb E\left[\beta\,\operatorname{stopgrad}\left(\log\pi_\theta(a\mid s)+\mathcal H_*\right)\right],\quad \beta=\log\alpha.
\]

Its gradient with respect to \(\beta\) is

\[
\nabla_\beta\mathcal L_\alpha=-\mathbb E[\log\pi+\mathcal H_*]=\mathcal H_\pi-\mathcal H_*.
\]

If the policy entropy falls below the target, this gradient is negative and gradient descent increases \(\log\alpha\), strengthening the entropy term. If entropy exceeds the target, the update tends to lower \(\alpha\). This implementation explicitly uses a **positive** target-entropy convention (\(0.5\) nats per action dimension); codebases that name a negative target-entropy value use different but equivalent sign conventions in their loss expression.

## 7. Full update cycle

1. Collect experience using uniform random actions during warm-up, then the stochastic actor. Store \((s,a,r,s',d)\) in replay memory.
2. Sample a minibatch from replay. Draw next actions from the actor and form soft Bellman targets using the target critics and current \(\alpha\).
3. Update both critics by minimizing their Bellman residuals.
4. Draw current actions with the reparameterized actor. Minimize \(\alpha\log\pi-\min(Q_1,Q_2)\) using gradients through the actions and critic input.
5. Update \(\log\alpha\) toward the target entropy.
6. Polyak-average both target critics.

The supplied `sac_numpy.py` uses two-hidden-layer ReLU MLPs, hand-coded NumPy backpropagation, Adam, a replay buffer, twin critics, target critics, a tanh-Gaussian actor, and automatic temperature tuning. There is no PyTorch, TensorFlow, Gym, or other RL library; only NumPy and the Python standard library are used.

## 8. Custom-environment training result

The environment is a one-dimensional damped point mass. Its continuous action is acceleration in \([-1,1]\), and the reward penalizes position error, velocity, and control effort:

\[
r_t=-\left(x_t^2+0.08v_t^2+0.01a_t^2\right).
\]

It starts at randomized position and velocity, runs for up to 100 steps, and terminates early if it moves outside the allowed range. The script was trained for 6,000 environment steps with seed 7. Evaluation uses 100 deterministic-policy episodes and the random baseline uses the same set of initial states.

| Evaluation | Mean return | Standard deviation |
|---|---:|---:|
| Uniform-random actions | -98.383 | 90.230 |
| Trained SAC policy | -6.996 | 6.559 |

Because reward is negative cost, a return closer to zero is better. The learned controller also had mean final absolute position \(0.0311\), with all 100 evaluation episodes reaching the 100-step horizon. This demonstrates learning on this small toy problem, not general-purpose control performance; it is one seed and one simple deterministic environment.

## 9. Short note: entropy and free will in human–AI philosophy

SAC’s entropy term gives an agent a mathematical reason not to collapse too early onto one action: among policies with similar expected return, it can prefer a broader distribution of actions. This resembles one narrow aspect of debates about agency—the availability of alternatives—but the analogy stops there. Randomness is not the same as authorship, deliberation, or acting for reasons. In SAC, action stochasticity is a parameterized distribution optimized against a specified reward and entropy target; its randomness does not by itself establish autonomy, consciousness, or metaphysical free will. Conversely, a deterministic policy is not automatically “unfree”: it may encode a sophisticated state-dependent control law. For human–AI comparisons, entropy is best treated as a measurable property of a policy’s action distribution, useful for exploration and robustness, rather than as a proxy for genuine freedom. The philosophical question is not simply whether alternatives are sampled, but what grounds the system’s goals, its capacity to respond to reasons, and the sense in which its choices are attributable to it.

## Reproduce

```bash
python sac_numpy.py --steps 6000 --seed 7 --out-dir .
```

Requires Python and NumPy. The run writes `sac_training_log.csv`, `sac_training_results.json`, and `sac_numpy_checkpoint.npz` to the chosen output directory. The saved weights can be evaluated without retraining with `python sac_numpy.py --eval-checkpoint sac_numpy_checkpoint.npz --episodes 100`. The checkpoint stores network parameters and temperature, but not Adam optimizer moments, so it is intended for inference/evaluation rather than an exact optimizer-state resume.
