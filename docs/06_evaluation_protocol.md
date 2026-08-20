# Evaluation Protocol

To prove this tool is academically and technically valid, you cannot simply run it on a dataset and say "look at these graphs." You must evaluate the *accuracy of the tool itself* using Ground Truth.

## 1. Synthetic Data Generation (The Ground Truth)
We must construct synthetic datasets using known Causal DAGs (Directed Acyclic Graphs). Because we generate the data, we know exactly where the bias is.

### Scenario A: The Null Hypothesis (Negative Control)
* **Setup**: Generate 100,000 rows. Feature values and missingness are assigned purely at random, entirely independent of the `Group` attribute.
* **Test Criteria**: The tool MUST NOT return any High or Critical findings. It must correctly suppress false positives via BH-FDR correction and effect-size filtering.

### Scenario B: Injected MNAR (Missing Not At Random)
* **Setup**: 
  * `Group` $\in \{A, B\}$. 
  * `Income` feature generated normally. 
  * If `Group == B`, forcibly delete 30% of the `Income` values.
* **Test Criteria**: The tool MUST flag `Income` missingness as High/Critical severity and associate it specifically with `Group == B`.

### Scenario C: The Large $N$ Small Effect Trap
* **Setup**: Generate 1,000,000 rows. Make the mean of `Feature_X` for Group A $= 10.0$ and Group B $= 10.01$. 
* **Test Criteria**: The $p$-value will be extremely significant ($< 0.0001$), but the tool MUST classify this as **Informational** because the Effect Size is trivial. This proves the tool is robust.

## 2. Real-World Benchmarks
After validating against synthetic data, run the tool against standard fairness benchmarks:

* **Adult Census Income**: 
  * *Expected Finding*: Strong label disparity based on Gender. Strong correlation between Occupation and Gender (Proxy variable detection).
* **COMPAS (ProPublica)**: 
  * *Expected Finding*: Disparate distributions in prior arrests and risk scores across Race.
* **German Credit**:
  * *Expected Finding*: Representation imbalances across Age brackets.
