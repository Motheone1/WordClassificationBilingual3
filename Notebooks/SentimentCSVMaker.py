import pandas as pd

# Load the input CSV
df = pd.read_csv("data/train.csv", encoding='utf-8')

# Ensure the column names are correct
# Rename if needed
if 'text' not in df.columns or 'sentiment' not in df.columns:
    print("❗ The expected columns 'text' and 'sentiment' were not found.")
    print("Detected columns:", df.columns.tolist())
    # Optional: Try to guess columns or raise error
    raise ValueError("Input CSV must have 'text' and 'sentiment' columns.")

# Filter by sentiment
positive_df = df[df['sentiment'].str.lower() == 'positive'][['text', 'sentiment']]
negative_df = df[df['sentiment'].str.lower() == 'negative'][['text', 'sentiment']]
neutral_df  = df[df['sentiment'].str.lower() == 'neutral'][['text', 'sentiment']]

# Save to separate CSVs
positive_df.to_csv("data/positive.csv", index=False)
negative_df.to_csv("data/negative.csv", index=False)
neutral_df.to_csv("data/neutral.csv", index=False)

print("✅ Files saved: positive.csv, negative.csv, neutral.csv")
import pandas as pd

# Define CSV to TXT mapping
files = {
    "positive.csv": "positive.txt",
    "negative.csv": "negative.txt",
    "neutral.csv": "neutral.txt"
}

for csv_file, txt_file in files.items():
    # Load CSV
    df = pd.read_csv(csv_file)

    # Ensure 'text' column exists
    if 'text' not in df.columns:
        raise ValueError(f"'text' column not found in {csv_file}")

    # Extract and write text column
    df['text'].to_csv(txt_file, index=False, header=False)

    print(f"✅ Extracted messages from {csv_file} into {txt_file}")
