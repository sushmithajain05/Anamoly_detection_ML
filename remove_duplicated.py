import pandas as pd

input_file = "anomalies_detected_3.csv" 
output_file = "cleaned_dataset.csv"

chunk_size = 100000  

with pd.read_csv(input_file, chunksize=chunk_size) as reader:
    for chunk in reader:
        chunk = chunk.drop_duplicates(subset=['id'])  
        
        chunk.to_csv(output_file, mode='a', index=False, header=not bool(pd.read_csv(output_file, nrows=1).empty))

print("Duplicate removal complete. Cleaned dataset saved as", output_file)