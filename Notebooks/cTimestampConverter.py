import pandas as pd
from datetime import datetime

# Load the CSV
df = pd.read_csv('Cosmos.csv - cosmos_data.csv')

# Assume the column with the timestamp is named 'UnixTimestamp'
# If the column name is different, change it below
df['TimeStampFixed'] = pd.to_datetime(df['_ts'], unit='s')

# Optional: Format as yyyy-mm-dd (string) if you don't want datetime objects
# df['ConvertedDate'] = df['ConvertedDate'].dt.strftime('%Y-%m-%d')

# Save to a new CSV file
df.to_csv('cosmos_data211.csv', index=False)

print("Conversion complete. Output saved to cosmos_data_converted.csv.")
