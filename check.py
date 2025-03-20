import pandas as pd

df1 = pd.read_csv('anomalies.csv')
df2 = pd.read_csv('anomalies_svm.csv')
df3 = pd.read_csv('anomalies_lof.csv')

count_df1 = df1.shape[0]
count_df2 = df2.shape[0]
count_df3 = df3.shape[0]

print(f"Total rows in anomalies.csv: {count_df1}")
print(f"Total rows in anomalies_svm.csv: {count_df2}")
print(f"Total rows in anomalies_lof.csv: {count_df3}")


total_rows = count_df1 + count_df2 + count_df3
print(f"\nTotal rows across all three datasets: {total_rows}")

common_rows = pd.merge(pd.merge(df1, df2, on=['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'frequency', 'kwh_imp']),
                       df3, on=['watts_total', 'pf_avg', 'vln_avg', 'amps_avg', 'frequency', 'kwh_imp'])


df1_diff = df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1)) & ~df1.apply(tuple, 1).isin(df3.apply(tuple, 1))]

df2_diff = df2[~df2.apply(tuple, 1).isin(df1.apply(tuple, 1)) & ~df2.apply(tuple, 1).isin(df3.apply(tuple, 1))]

df3_diff = df3[~df3.apply(tuple, 1).isin(df1.apply(tuple, 1)) & ~df3.apply(tuple, 1).isin(df2.apply(tuple, 1))]

different_rows = pd.concat([df1_diff, df2_diff, df3_diff])

print(f"Total number of common rows: {common_rows.shape[0]}")
print(f"Total number of different rows: {different_rows.shape[0]}")

common_rows.to_csv('common_rows.csv', index=False)
different_rows.to_csv('different_rows.csv', index=False)

print("\nCommon Rows (first 5 rows):")
print(common_rows.head())

print("\nDifferent Rows (first 5 rows):")
print(different_rows.head())
