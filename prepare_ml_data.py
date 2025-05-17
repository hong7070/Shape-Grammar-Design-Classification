import os
import ast
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV     = os.path.join(BASE_DIR, "processed_responses.csv")
X_WO_CONF_CSV = os.path.join(BASE_DIR, "X_wo_conf.csv")
X_WITH_CONF_CSV = os.path.join(BASE_DIR, "X_with_conf.csv")
Y_CSV         = os.path.join(BASE_DIR, "y.csv")

# 1) Load processed responses
df = pd.read_csv(INPUT_CSV)

# 2) Parse the "Responses" string into list of length 22
# Each row: 20 sequence elements + Label + Confidence
parsed = df["Responses"].apply(ast.literal_eval)
seqs   = parsed.apply(lambda x: x[:20])
labels = parsed.apply(lambda x: x[20])
confs  = parsed.apply(lambda x: int(x[21]))

# 3) One‐hot encode the 20 sequence positions
# We treat each position as a separate categorical column.
# So we'll transform seqs (n_samples x 20) → a big one‐hot matrix.

# Prepare the encoder to cover all possible tokens:
# digits '0','1','2','3','4' and letters 'A','B','C','D'
tokens = [['0','1','2','3','4','A','B','C','D']] * 20
ohe = OneHotEncoder(categories=tokens, sparse=False, dtype=int, handle_unknown='ignore')

X_seq = ohe.fit_transform(seqs.tolist())
# Feature names like "pos0_0", "pos0_1", ..., "pos19_D"
ohe_feature_names = []
for pos in range(20):
    for val in ohe.categories_[pos]:
        ohe_feature_names.append(f"pos{pos}_{val}")

X_seq_df = pd.DataFrame(X_seq, columns=ohe_feature_names)

# 4) Build two feature sets
# A) Without confidence
X_wo_conf = X_seq_df.copy()

# B) With confidence
X_with_conf = X_seq_df.copy()
X_with_conf["confidence"] = confs.values

# 5) One-hot encode the labels
label_ohe = OneHotEncoder(sparse=False, dtype=int)
y = label_ohe.fit_transform(labels.values.reshape(-1,1))
label_classes = label_ohe.categories_[0].tolist()
y_df = pd.DataFrame(y, columns=[f"label_{c}" for c in label_classes])

# 6) Save to CSV
X_wo_conf.to_csv(X_WO_CONF_CSV, index=False)
X_with_conf.to_csv(X_WITH_CONF_CSV, index=False)
y_df.to_csv(Y_CSV, index=False)

print(f"✔ X (no conf) → {X_WO_CONF_CSV}")
print(f"✔ X (with conf) → {X_WITH_CONF_CSV}")
print(f"✔ y            → {Y_CSV}")
print(f"✔ Classes: {label_classes}")
