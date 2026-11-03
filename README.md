# Codveda Technology — Machine Learning Internship

**Intern:** Omar Ahmed Mohamed Fathy Fayed — Machine Learning Intern
**Tasks:** all 9 tasks of the Machine Learning task list (3 per level), each as a fully executed Jupyter notebook with plots,
metrics and written conclusions.

## Repository structure

```
├── data/                                   # datasets provided by Codveda
│   ├── iris.csv
│   ├── stock_prices.csv
│   ├── house_prediction.csv
│   ├── churn-bigml-80.csv / churn-bigml-20.csv
│   └── sentiment.csv
├── Level_1_Basic/
│   ├── Task1_Data_Preprocessing/      data_preprocessing.ipynb
│   ├── Task2_Linear_Regression/       linear_regression.ipynb
│   └── Task3_KNN_Classifier/          knn_classifier.ipynb
├── Level_2_Intermediate/
│   ├── Task1_Logistic_Regression/     logistic_regression.ipynb
│   ├── Task2_Decision_Tree/           decision_tree.ipynb
│   └── Task3_KMeans_Clustering/       kmeans_clustering.ipynb
├── Level_3_Advanced/
│   ├── Task1_Random_Forest/           random_forest.ipynb
│   ├── Task2_SVM/                     svm_classification.ipynb
│   └── Task3_Neural_Network/          neural_network.ipynb
└── requirements.txt
```
Every task folder also has an `images/` folder with the saved figures (and `output/` where a model or processed data is saved).

## Results at a glance

### Level 1 — Basic
| Task | Dataset | Key result |
|---|---|---|
| **1. Data preprocessing** | Stock prices (497k rows, 505 companies) | 27 missing values handled (non-trading rows dropped, gaps filled with same-day close — shown to beat median fill); one-hot (weekday) + label encoding (505 symbols); StandardScaler fit on train only; chronological train/test split; reusable `ColumnTransformer` saved. |
| **2. Linear regression** | Boston housing (506 rows) | Multiple LR: **test R² = 0.758, MSE = 12.4** (5-fold CV R² 0.752). Simple LR on rooms: +$8.2k per extra room. Biggest drivers: `LSTAT` (−), `RM` (+), `DIS` (−). |
| **3. KNN classifier** | Iris | K = 1…30 compared with 10-fold CV; selected **K = 11 → test accuracy 95.6 %**, macro precision 0.96 / recall 0.96. Decision-region plots for small vs large K. |

### Level 2 — Intermediate
| Task | Dataset | Key result |
|---|---|---|
| **1. Logistic regression** | Telecom churn | Balanced LR: **recall 0.77, ROC-AUC 0.82**. Odds ratios: international plan ×8.6, +1 SD day minutes ×2.0, +1 SD service calls ×2.0, voice-mail plan ×0.24. |
| **2. Decision tree** | Iris | Unpruned tree (8 leaves) overfits; pruned tree (`max_depth = 3`, chosen by CV) → **test accuracy 97.8 %, F1 0.978**. Cost-complexity pruning path also shown. |
| **3. K-Means clustering** | Stock prices → 499 companies | Elbow at **k = 5**: steady compounders, distressed/energy, high-growth (FB, AMZN, NFLX, NVDA), mega-cap blue chips (AAPL, MSFT, JPM), under-performers. |

### Level 3 — Advanced
| Task | Dataset | Key result |
|---|---|---|
| **1. Random forest** | Telecom churn | Randomized search (40 × 5-fold CV) → **test F1 0.863, accuracy 96.3 %, recall 0.83, AUC 0.935**. Top features: day minutes, customer-service calls, international plan. |
| **2. SVM** | Telecom churn | Linear vs RBF vs polynomial kernels: **RBF best (test AUC 0.918** vs 0.832 linear). 2-D decision boundaries visualised. |
| **3. Neural network (Keras)** | Telecom churn | 64-32-16 ReLU network with batch-norm, dropout, L2, class weights and early stopping → **test accuracy 89.8 %, AUC 0.925, recall 0.80**. Training/validation loss curves + overfitting comparison. |

### Churn model comparison (same test set, 667 customers)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression (balanced) | 0.774 | 0.361 | 0.768 | 0.492 | 0.819 |
| SVM — RBF kernel | 0.879 | 0.554 | 0.758 | 0.640 | 0.918 |
| Neural network | 0.898 | 0.608 | 0.800 | 0.691 | 0.925 |
| **Random forest (tuned)** | **0.963** | **0.898** | **0.832** | **0.863** | **0.935** |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```
Open any notebook and run all cells — data paths are relative (`../../data/`), so keep the folder structure as is.
All random seeds are fixed (`random_state = 42`), so results are reproducible.

## Tools
Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · matplotlib · seaborn · Jupyter

#CodvedaJourney #CodvedaExperience #FutureWithCodveda
