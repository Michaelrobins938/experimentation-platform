# Modern Experimentation: Variance Reduction & Sequential Monitoring
## Integrating CUPED, Sequential SPRT, and Bayesian Sensitivity analysis

**Technical Whitepaper v1.0.0**

| **Attribute** | **Value** |
|---|---|
| **Version** | 1.0.0 |
| **Status** | Production-Ready |
| **Date** | January 31, 2026 |
| **Classification** | Statistical Methodology |
| **Document Type** | Technical Whitepaper |

---

## **Abstract**

Standard A/B testing methods are often slow (high sample size requirements) or risky (prone to "p-hacking" via early peeking). This whitepaper details a **High-Velocity Experimentation Framework** built to overcome these limitations. We combine **CUPED** for pre-experiment variance reduction with **Sequential Probability Ratio Tests (SPRT)** to allow for safe, real-time monitoring of experiment results without inflating False Positive Rates (FPR).

---

## **Glossary & Notation**

| **Term** | **Definition** |
|---|---|
| **CUPED** | Controlled-experiment Using Pre-Experiment Data; a technique to reduce variance using historical information. |
| **SPRT** | Sequential Probability Ratio Test; a statistical test that allows for "peeking" at data as it arrives. |
| **MDE** | Minimum Detectable Effect; the smallest effect size an experiment is powered to detect. |
| **FWER** | Family-Wise Error Rate; the probability of making at least one Type I error. |
| **Power** | The probability of correctly rejecting a false null hypothesis (target: 80%). |

---

## **1. The Efficiency Problem in A/B Testing**

Most organizations face a trade-off between **Velocity** (running many tests quickly) and **Integrity** (ensuring statistical rigor). Standard T-tests require fixed sample sizes and strictly prohibit peeking. Our framework solves this via two primary mechanisms:
1. **Variance Reduction** (CUPED) to shrink the required sample size.
2. **Sequential Monitoring** (SPRT) to stop experiments the moment significance is reached.

---

## **2. Variance Reduction via CUPED**

CUPED utilizes pre-experiment data to "denoise" the metric of interest. By adjusting the metric $Y$ using its historical value $X$:

$$Y_{adj} = Y - \theta(X - E[X])$$

The optimal $\theta = \frac{Cov(X, Y)}{Var(X)}$ minimizes the variance of the treatment effect estimate. In practice, this can reduce variance by **30-50%**, effectively doubling the "speed" of the experimentation platform.

---

## **3. Sequential Probability Ratio Test (SPRT)**

To allow stakeholders to monitor experiments in real-time without the "peeking problem," we implement the **Wald Sequential Test**. The Likelihood Ratio for $n$ observations is:

$$LR_n = \prod_{i=1}^n \frac{f(x_i | H_1)}{f(x_i | H_0)}$$

We define two boundaries:
- **Upper Bound (A):** Reject $H_0$ (Success).
- **Lower Bound (B):** Accept $H_0$ (Insignificant).

The experiment continues as long as $B < LR_n < A$. This ensures that the Type I error is controlled at $\alpha$ regardless of how many times the data is peeked.

---

## **4. Bayesian Sensitivity Analysis**

For critical business decisions, we augment frequentist p-values with **Bayesian Posterior Probabilities**.
- **Prior:** Weakly informative or based on historical "win rates."
- **Posterior:** $P(\text{Lift} > 0 | \text{Data})$.
- **ROPE (Region of Practical Equivalence):** Defining a threshold below which an effect is considered "zero" for business purposes.

---

## **5. Guardrail & Secondary Metrics**

A "winning" experiment on the primary metric (e.g., Conversion) can still be a failure if it damages secondary "guardrail" metrics (e.g., Latency, Unsubscribes). Our platform automatically calculates:
- **Interaction Effects:** Determining if multiple concurrent tests are interfering with one another.
- **Sample Ratio Mismatch (SRM):** A Chi-squared test to detect assignment bias or engineering bugs in the randomization logic.

---

## **6. Technical Implementation Specification**

- **Framework:** React/Next.js for the Command Center.
- **Stats Engine:** Python (NumPy, SciPy, Statsmodels).
- **Architecture:** Microservices-based, with a central "Validator" service that ensures every test is properly powered before launch.

---

## **7. Causal Interpretation & Limitations**

- **External Validity:** Results might not generalize to other time periods or segments.
- **Carryover Effects:** Users in one experiment might have lingering effects that influence future tests.
- **Network Effects:** In two-sided marketplaces, treatment may leak to the control group.

---

## **8. Conclusion**

By shifting from static A/B testing to dynamic, sequence-aware experimentation, companies can drastically increase their "learning velocity" while maintaining strict mathematical defensibility. This platform is optimized for organizations that value data-driven agility over traditional, slow-moving statistical cycles.
