import pandas as pd

input_file = "studer.csv"  
chunk_size = 1000000  

for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
    output_file = f"split_part_{i+1}.csv"
    chunk.to_csv(output_file, index=False)
    print(f"Saved {output_file}") 