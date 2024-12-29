import pandas as pd
import matplotlib.pyplot as plt

# Ensure reproducible styling
plt.style.use('seaborn-v0_8-dark')

# Load the data
data = pd.read_csv('meteorite_data.csv')
data = data.drop_duplicates(subset=['name', 'year', 'reclat', 'reclong'])
data = data.dropna()

# Step 1: Raw Data Visualization
plt.figure(figsize=(8, 6))
plt.scatter(data['reclong'], data['reclat'], alpha=0.5, color='gray')
plt.title('Raw Meteorite Data', fontsize=16)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.grid(True)
plt.savefig('step1_raw_data.png')
plt.close()

# Step 2: Trend Analysis Visualization
data['year'] = data['year'].astype(int)
data_filtered = data[data['year'] > 1800]
landings_by_year = data_filtered.groupby('year')['name'].count()

max_year = landings_by_year.idxmax()
max_value = landings_by_year.max()

plt.style.use('dark_background')
print(plt.style.available)

plt.figure(figsize=(12, 8))
plt.plot(landings_by_year.index, landings_by_year.values, label='Meteorite Landings Over Time', linewidth = 3)
plt.xlabel('Year')
plt.ylabel('Number of Landings')
plt.title('Trend Analysis: Meteorite Landings', fontweight='bold')
plt.grid(True)
plt.legend()
plt.savefig('step2_trend_analysis.png')
plt.close()