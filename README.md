##  Covid-ICU-Risk-Predictor
Description: Machine learning system predicting ICU admission risk for COVID-19 patients using PySpark and cluster-based modeling. Features include data cleaning of 200K+ clinical records, SMOTE balancing, K-Means clustering, and comparative evaluation of logistic regression models. Achieved 0.77 AUC with 58.2% recall.


## 🔍 Project Overview
Developed a machine learning system to predict ICU admission risk for COVID-19 patients using clinical data (200,031 records, 21 features). Key innovations:
- **PySpark-powered data pipeline** handling missing values (encoded as "?") and outliers
- **Cluster-based modeling** with K-Means segmentation (2 clusters with 4.3% vs 12.6% ICU risk)
- **SMOTE-balanced classifiers** addressing extreme class imbalance (8.7% ICU cases)
- **Comparative model evaluation** showing 22% F1-score improvement over baseline

## 📊 Key Results
| Model | AUC | Recall | F1-Score | Improvement |
|-------|-----|--------|----------|-------------|
| Baseline (Unbalanced) | 0.761 | 0.52 | 0.31 | - |
| Cluster-Based (K-Means) | 0.765 | 0.519 | 0.329 | +6.1% |
| **Balanced Cluster (SMOTE)** | **0.772** | **0.582** | **0.382** | **+22.0%** |

*Top ICU risk factors: Age > 60, Pneumonia, Intubation Status*

## 🛠️ Technical Implementation
### Data Pipeline
python
# PySpark preprocessing example
from pyspark.ml.feature import Imputer, StandardScaler

imputer = Imputer(inputCols=features, outputCols=features)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")


### Modeling Architecture
1. **K-Means Clustering** (k=2 via Elbow Method)
2. **Cluster-specific classifiers** with logistic regression
3. **SMOTE balancing** applied per cluster
4. **Aggregated predictions** with PySpark MLlib

### Critical Libraries
- `PySpark` (distributed data processing)
- `Scikit-learn` (logistic regression, SMOTE)
- `Seaborn/Matplotlib` (visualization)
- `Pandas` (data manipulation)
