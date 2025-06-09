COVID-ICU-Risk-Prediction  

Description: Machine learning system predicting ICU admission risk for COVID-19 patients using PySpark and cluster-based modeling. Features include data cleaning of 200K+ clinical records, SMOTE balancing, K-Means clustering, and comparative evaluation of logistic regression models. Achieved 0.77 AUC with 58.2% recall.


# COVID-19 ICU Admission Risk Prediction

![ICU Prediction Workflow](plots/workflow_diagram.png) *Figure: End-to-end predictive pipeline*

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

## 📂 Repository Structure

├── data_processing/
│   ├── data_cleaning.py        # Handling "?", normalization, outlier capping
│   └── pyspark_preprocessing.py  # Distributed imputation (200K+ records)
├── modeling/
│   ├── baseline_model.ipynb    # Logistic regression + evaluation
│   ├── clustering.py           # K-Means implementation (Elbow method)
│   └── cluster_classifiers.py  # Local models per cluster
├── evaluation/
│   ├── metrics.py              # AUC, F1-score, precision-recall
│   └── model_comparison.png    # Performance visualization
├── reports/
│   ├── Report.pdf              # Full technical report
│   └── presentation.pptx       # Summary deck
├── plots/                      # Key visualizations
│   ├── age_icu_distribution.png
│   ├── feature_correlation.png
│   └── cluster_analysis.png
├── requirements.txt            # Python dependencies
└── README.md


## 🚀 Getting Started
1. Install dependencies:
bash
pip install -r requirements.txt

2. Run data pipeline:
bash
python data_processing/pyspark_preprocessing.py

3. Train cluster-based model:
bash
python modeling/cluster_classifiers.py


## 📌 Key Insights from Analysis
1. **Age is critical predictor**: 60-79 age group has 4× higher ICU risk
2. **Comorbidity impact**: Pneumonia increases ICU risk by 37% vs baseline
3. **Data quality matters**: Capping age outliers (IQR method) preserved 3.12% critical cases
4. **Cluster differences**: 
   - Cluster 0: Younger patients with respiratory symptoms (4.3% ICU)
   - Cluster 1: Elderly with comorbidities (12.6% ICU)

## 🔗 References
1. Chawla, N.V. et al. (2002) - SMOTE sampling
2. Zaharia, M. et al. (2016) - Apache Spark
3. Arthur, D. & Vassilvitskii, S. (2007) - k-means++

*Dataset: Mexican COVID-19 Patient Data (200,031 records)*


---

### Key Visuals to Include in Repo  
1. *plots/workflow_diagram.png*: End-to-end system architecture
2. *plots/cluster_analysis.png*: K-Means clusters with ICU rates
3. *evaluation/model_comparison.png*: AUC/F1-score improvements
4. *plots/feature_correlation.png*: Heatmap of top ICU predictors
