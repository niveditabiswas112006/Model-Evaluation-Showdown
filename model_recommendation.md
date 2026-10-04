# Model Evaluation Showdown: Recommendation Report

## Dataset Context
To ensure the originality of this submission and avoid using commonly duplicated datasets like Iris, Titanic, or Breast Cancer, this evaluation uses the **Red Wine Quality** dataset from the UCI Machine Learning Repository. 

The original dataset ranks wine quality on a scale of 0 to 10. For this classification task, it has been converted into a **binary classification problem**: predicting whether a wine is of "Good Quality" (quality score >= 6) or "Bad Quality" (quality score < 6) based on 11 physicochemical properties (e.g., acidity, sugar, pH, alcohol).

## Metrics Comparison Table

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7350   | 0.7547    | 0.7477 | 0.7512   |
| K-Nearest Neighbors | 0.7425   | 0.7511    | 0.7757 | 0.7632   |
| Decision Tree       | 0.7375   | 0.7633    | 0.7383 | 0.7506   |

### Confusion Matrices

**Logistic Regression**
| | Predicted Bad Quality (0) | Predicted Good Quality (1) |
|---|---|---|
| **Actual Bad Quality (0)** | 134 | 52 |
| **Actual Good Quality (1)** | 54 | 160 |

**K-Nearest Neighbors (KNN)**
| | Predicted Bad Quality (0) | Predicted Good Quality (1) |
|---|---|---|
| **Actual Bad Quality (0)** | 131 | 55 |
| **Actual Good Quality (1)** | 48 | 166 |

**Decision Tree**
| | Predicted Bad Quality (0) | Predicted Good Quality (1) |
|---|---|---|
| **Actual Bad Quality (0)** | 137 | 49 |
| **Actual Good Quality (1)** | 56 | 158 |

---

## Final Recommendation

### **Which model would you deploy?**
Based on this evaluation, I would recommend deploying the **K-Nearest Neighbors (KNN)** model for this specific wine quality prediction task.

**Justification:**
1. **Best Overall Performance:** KNN slightly edges out the competition with the highest Accuracy (74.25%) and the highest F1-Score (76.32%).
2. **Highest Recall:** KNN achieved the highest Recall (77.57%), correctly identifying 166 out of 214 "Good Quality" wines. If a winery is trying to identify potential premium wines for their catalog, ensuring they don't accidentally throw out or downgrade a great batch is highly beneficial.
3. **Non-Linear Relationships:** The physicochemical properties of wine (like the balance of alcohol, sugar, and acidity) do not always have a strict linear relationship with quality. As a non-parametric model, KNN can map out these complex, clustered relationships better than a simple linear boundary like Logistic Regression.

### **When would you choose differently?**
While KNN performs best here, there are distinct scenarios where the other models would be preferred:

*   **Choose Decision Tree if:** The winery's stakeholders (e.g., winemakers, sommeliers) want actionable, interpretable rules. A Decision Tree provides clear thresholds (e.g., "If alcohol > 11% and pH < 3.2, the wine is good"). This allows winemakers to physically adjust their brewing process, rather than just treating the AI as a black box. The Decision Tree also boasted the highest precision (76.33%) in our tests.
*   **Choose Logistic Regression if:** The dataset grew to millions of rows or the model needed to be deployed to a low-resource environment (like a mobile app or edge device). KNN requires the entire dataset to be loaded into memory to calculate distances for every single prediction, which scales poorly. Logistic Regression only stores a few feature weights, making inference virtually instantaneous.
