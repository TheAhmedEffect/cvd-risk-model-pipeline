# CVD Risk Classification Pipeline

A multi-class cardiovascular risk model built on a 1,529-record measurement dataset, taken
from preprocessing through to model explainability. Python, scikit-learn, LightGBM,
XGBoost, SHAP and LIME.

`CVD Dataset.csv` sits beside the notebook. Read the dataset section below before drawing
anything from the results — it is the most important thing in this repository.

## What the notebook does

- **Preprocessing as a pipeline, not ad hoc steps** — numeric features go through mean
  imputation and `StandardScaler`, categorical features through their own transformer,
  combined in a `ColumnTransformer` so the same transformation applies identically to any
  new data.
- **Feature selection** — `SelectKBest` on the processed matrix.
- **Split** — 80/20, stratified on the three-class target, fixed `random_state`.
- **Models** — logistic regression, Gaussian naive Bayes, decision tree, k-nearest
  neighbours, random forest, bagging, AdaBoost, gradient boosting, an MLP, an RBF-kernel
  SVM, LightGBM and XGBoost, plus a soft-voting ensemble and a stacking classifier with a
  logistic-regression meta-learner.
- **Class imbalance** — SMOTE over-sampling applied to the training set.
- **Tuning** — `GridSearchCV` over the tree ensembles.
- **Explainability** — SHAP `TreeExplainer` for global feature attribution and a LIME
  `LimeTabularExplainer` for individual predictions.

## The dataset is synthetic, and partly incoherent

This is the finding that governs every number below. The data does not describe real
patients, and its target label is not consistent with its own clinical features.

The evidence, all checkable in a few lines of pandas against the committed CSV:

- **The `LOW` risk class contradicts every feature that defines risk.** Grouped by the
  target, `LOW` has the *highest* mean systolic blood pressure (132.5, against 124.5 for
  both `HIGH` and `INTERMEDIARY`), the highest fasting blood sugar (123.9), the oldest mean
  age (48.5), and a mean `CVD Risk Score` of 17.0 — above `INTERMEDIARY` at 16.1. A
  lowest-risk group with the worst blood pressure and the oldest patients is not a
  medically possible cohort.
- **Binary risk factors sit at implausible prevalence.** Smoking is 789 Y to 740 N and
  diabetes 777 Y to 752 N — near coin-flips, where real population prevalence is nothing
  like 50%. Sex, family history and physical activity are all split with the same
  artificial evenness.
- **Continuous variables look uniformly drawn between round bounds.** Total cholesterol
  has mean 198.5 and standard deviation 57.8 across an exact 100–300 range; a
  uniform(100,300) distribution gives 200.0 and 57.7. Real cholesterol is not uniform.
  Weight stops exactly at 120.0, height at 2.00, abdominal circumference at 120.
- **Thirteen records carry a negative estimated LDL**, to −18 mg/dL. That is what Friedewald
  produces when total cholesterol and HDL are generated independently of each other rather
  than measured from the same person.
- **BMI is not weight divided by height squared.** It was generated separately rather than
  derived, unlike `Height (cm)` and `Waist-to-Height Ratio`, which are exact derivations.
- **Missing values are injected, not natural.** Every numeric column is missing 64–82 of
  1,529 values, a flat 4–5% everywhere. Real clinical missingness clusters by test.

## Results

Three-class classification on 153 held-out records.

| Model family | Accuracy | ROC-AUC | Macro recall |
|---|---|---|---|
| Best single model | 0.575 | 0.629 | 0.498 |
| Stacking / voting ensembles | 0.556 | 0.651 | 0.427 |
| Tuned random forest (GridSearchCV) | 0.503 | 0.619 | 0.383 |
| Weakest baselines | 0.516 | 0.627 | 0.416 |

## What the results actually say

**Predicting the majority class alone scores 0.476.** Against that baseline, the best model
here adds about ten accuracy points, and the tuned random forest adds nothing at all. That
is the honest reading of the table: this is a weak result, and the pipeline is not the
reason.

The near-zero recall on the `LOW` class in the notebook output is usually explained by
class imbalance, and that explanation is wrong here. `LOW` is unlearnable because its
features look *worse* than `INTERMEDIARY`'s on blood pressure, blood sugar and age. No
classifier can separate a class whose inputs point the opposite way to its label. Only the
`HIGH` class carries any usable relationship — smoking 62.5% against 39.8% for
`INTERMEDIARY`, diabetes 61.3% against 39.9%, BMI 29.9 against 27.0 — which is why the
models recover roughly that much and no more.

Added capacity could not fix this, and did not. The stacking and voting ensembles do not
beat the better single models by any margin worth claiming, and the tuned random forest
lands below both. When twelve model families and two ensembles all converge just above the
majority-class baseline, the ceiling belongs to the data.

## Known limitations of the pipeline itself

- The `ColumnTransformer` is fitted on the full dataset before the train/test split, so
  imputation means and scaling statistics are computed over the test rows too. The leak is
  small for mean-and-scale preprocessing, but the fit belongs inside the split.
- The LIME explainer is configured with binary class names against a three-class target,
  so its per-instance labels are mislabelled.
- No cross-validated confidence intervals, so differences of a few points between these
  models are not distinguishable from noise.
