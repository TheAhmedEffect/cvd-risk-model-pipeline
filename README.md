# CVD Risk Classification Pipeline

A multi-class cardiovascular risk model built on a 1,529-record clinical measurement
dataset, taken from preprocessing through to model explainability. Python, scikit-learn,
LightGBM, XGBoost, SHAP and LIME.

The dataset is **not included** in this repository. The notebook expects a
`CVD Dataset.csv` beside it, with 1,529 rows and columns covering sex, age, weight,
height, BMI, abdominal circumference, blood pressure, total cholesterol, HDL, estimated
LDL, fasting blood sugar, smoking status, diabetes status, physical activity level and
family history, against a three-level CVD risk target.

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

## Results

Three-class risk classification on 153 held-out records. Accuracy and macro recall across
the models tried:

| Model family | Accuracy | ROC-AUC | Macro recall |
|---|---|---|---|
| Best single model | 0.575 | 0.629 | 0.498 |
| Stacking / voting ensembles | 0.556 | 0.651 | 0.427 |
| Tuned random forest (GridSearchCV) | 0.503 | 0.619 | 0.383 |
| Weakest baselines | 0.516 | 0.627 | 0.416 |

## What the results actually say

The honest headline is that none of these models works well. Accuracy sits around
0.50–0.58 on a three-class problem and ROC-AUC around 0.62–0.65, so there is real but
weak signal. Added model capacity did not help: the stacking and voting ensembles did not
beat the better single models by any margin worth claiming, and the tuned random forest
came out below both. On 1,529 records with this feature set, that pattern points at the
data rather than the model choice.

The macro recall numbers are the ones to read. They are consistently far below accuracy,
and the lowest-risk class is sometimes recalled at or near zero — the models lean on the
majority class, which accuracy rewards and macro recall exposes.

## Known limitations

- The `ColumnTransformer` is fitted on the full dataset before the train/test split, so
  imputation means and scaling statistics are computed over the test rows too. The leak is
  small for mean-and-scale preprocessing but it is a leak, and the fit belongs inside the
  split.
- The LIME explainer is configured with binary class names against a three-class target,
  so its per-instance labels are mislabelled.
- No cross-validated confidence intervals, so differences of a few points between these
  models are not distinguishable from noise.
