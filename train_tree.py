import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load features & labels
# One‐hot sequence only
X_seq_only = pd.read_csv("X_wo_conf.csv")

# Sequence + confidence
X_with_conf = pd.read_csv("X_with_conf.csv")

# Hard labels
y = pd.read_csv("y.csv").idxmax(axis=1)

# Confidence (0–10) for weighting
conf = X_with_conf["confidence"]

# Train/test split
X_train_seq, X_test_seq, y_train, y_test, conf_train, conf_test = train_test_split(
    X_seq_only, y, conf, test_size=0.30, random_state=42, stratify=y
)

X_train_conf, X_test_conf, _, _, _, _ = train_test_split(
    X_with_conf, y, conf, test_size=0.30, random_state=42, stratify=y
)

# Prepare sample weights
# Normalize confidence to [0,1] for sample_weight
sample_weights = conf_train.values / 10.0

# Decision Tree WITHOUT confidence (unweighted)
dt_wo = DecisionTreeClassifier(random_state=42)
dt_wo.fit(X_train_seq, y_train)  
pred_wo_dt = dt_wo.predict(X_test_seq)
print("DT without confidence:")
print(" Accuracy:", accuracy_score(y_test, pred_wo_dt))
print(classification_report(y_test, pred_wo_dt))

# Decision Tree WITH confidence as feature (unweighted)
dt_wf = DecisionTreeClassifier(random_state=42)
dt_wf.fit(X_train_conf, y_train)  
pred_wf_dt = dt_wf.predict(X_test_conf)
print("DT with confidence as feature:")
print(" Accuracy:", accuracy_score(y_test, pred_wf_dt))
print(classification_report(y_test, pred_wf_dt))

# Decision Tree WITH sample weights (sequence only)
dt_sw = DecisionTreeClassifier(random_state=42)
dt_sw.fit(X_train_seq, y_train, sample_weight=sample_weights)
pred_sw_dt = dt_sw.predict(X_test_seq)
print("DT with sequence + sample weights:")
print(" Accuracy:", accuracy_score(y_test, pred_sw_dt))
print(classification_report(y_test, pred_sw_dt))

# Random Forest WITHOUT confidence
rf_wo = RandomForestClassifier(n_estimators=100, random_state=42)
rf_wo.fit(X_train_seq, y_train)
pred_wo_rf = rf_wo.predict(X_test_seq)
print("RF without confidence:")
print(" Accuracy:", accuracy_score(y_test, pred_wo_rf))
print(classification_report(y_test, pred_wo_rf))

# Random Forest WITH confidence as feature
rf_wf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_wf.fit(X_train_conf, y_train)
pred_wf_rf = rf_wf.predict(X_test_conf)
print("RF with confidence as feature:")
print(" Accuracy:", accuracy_score(y_test, pred_wf_rf))
print(classification_report(y_test, pred_wf_rf))

# Random Forest WITH sample weights
rf_sw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sw.fit(X_train_seq, y_train, sample_weight=sample_weights)
pred_sw_rf = rf_sw.predict(X_test_seq)
print("RF with sequence + sample weights:")
print(" Accuracy:", accuracy_score(y_test, pred_sw_rf))
print(classification_report(y_test, pred_sw_rf))