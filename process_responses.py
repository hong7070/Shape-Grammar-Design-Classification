import os
import re
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DESIGN_CSV = os.path.join(BASE_DIR, "design_sequence.csv")
SURVEY_CSV = os.path.join(BASE_DIR, "Design Labeling Survey (Responses).csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "processed_responses.csv")

# 1) Load your design sequences
design_df = pd.read_csv(DESIGN_CSV)

# 2) Load the raw survey responses
survey_df = pd.read_csv(SURVEY_CSV)

rows = []
# 3) Iterate each response row
for _, resp in survey_df.iterrows():
    # For every "Design Label for File: XXXX.jpg" column (with optional .1/.2 suffix)
    for col in survey_df.columns:
        m = re.match(r"^Design Label for File:\s*([0-9]{4}\.jpg)(?:\.\d+)?$", col)
        if not m:
            continue
        file_name = m.group(1)
        label     = resp[col]
        if pd.isna(label):
            continue

        # Find matching confidence column (also allowing .1/.2 suffix)
        conf_pattern = rf"^Confidence Level for {re.escape(file_name)}(?:\.\d+)?$"
        conf_cols = [c for c in survey_df.columns if re.match(conf_pattern, c)]
        if not conf_cols:
            continue
        confidence = resp[conf_cols[0]]
        if pd.isna(confidence):
            continue

        # 4) Lookup the 20-element sequence
        seqs = design_df.loc[design_df["File name"] == file_name, "Design Sequence"]
        if seqs.empty:
            print(f"Warning: no sequence found for {file_name}")
            continue
        seq = seqs.iloc[0].strip()
        # strip trailing ']' so we can append
        if seq.endswith("]"):
            seq = seq[:-1]

        # 5) Build the final array-string
        result = f"{seq}, \"{label}\", '{int(confidence)}']"

        rows.append({
            "File name": file_name,
            "Responses": result
        })

# 6) Assemble into a DataFrame
out_df = pd.DataFrame(rows, columns=["File name", "Responses"])

# 7) Sort by numeric file index (0001.jpg → 1, etc.)
out_df["num"] = out_df["File name"].str.replace(r"\.jpg$", "", regex=True).astype(int)
out_df = out_df.sort_values("num").drop(columns="num")

# 8) Write out the final CSV
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"✔ Processed {len(out_df)} total response rows")
print(f"✔ Written to: {OUTPUT_CSV}")
