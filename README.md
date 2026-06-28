# Incident Duration Prediction: A Machine Learning Post-Mortem

This project explores an end-to-end machine learning pipeline built to predict IT/Network incident resolution times (`duration_minutes`). While many data science projects focus on clean, synthetic success stories, this repository documents a real-world engineering challenge: **identifying a complete lack of statistical signal in an operational dataset and pivoting to find a solution.**

## 📊 The Challenge & Initial Models
The goal was to predict exact resolution times using features like raw severity, estimated revenue loss, and regional impact scores. 
* **Linear Regression Baseline:** $R^2 = -0.0021$
* **Random Forest Regressor (100 trees, depth 10):** $R^2 = -0.0022$

When both linear and non-linear models converge on a near-zero or negative $R^2$, it indicates the model is simply guessing the flat average—signaling zero correlation between the features and the target variable.

## 👁️ The Human Eye Audit (Operational Paradox)
Upon inspecting the raw data side-by-side, a massive operational paradox was uncovered:
* **Critical Tickets** (High revenue loss) were resolved rapidly (e.g., 30–60 mins) due to emergency "all-hands-on-deck" escalation workflows.
* **Low Tickets** sat in queues unassigned, racking up massive durations (e.g., 500+ mins) despite having zero business impact.

Because standard regression models expect higher severity to equal longer durations, the models were completely blinded by this real-world queuing behavior.

## 🔄 The Pivots
1. **Feature Engineering:** Integrated `service_impact_score`, `region_impact_score`, and `ticket_count` to provide operational context. The regression still failed, confirming the exact minute metric was pure noise.
2. **Classification Pivot:** Converted the objective from predicting minutes to predicting an **SLA Breach (>180 Mins)** using a `RandomForestClassifier`.
3. **Handling Class Imbalance:** The initial classifier hit 75% accuracy by lazily guessing "Delayed" for every row. By implementing `class_weight='balanced'`, I penalized the model, forcing it to play fair and breaking the Class 0 (On-Time) precision out of absolute zero.

## 💡 Key Takeaway
This project perfectly demonstrates that a machine learning pipeline is only as good as its underlying data context. In a production environment, this analysis serves as the data-backed justification needed to halt deployment and request deeper operational telemetry (such as technician availability, sub-component asset tags, or true root-cause categories).
