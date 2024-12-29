import pandas as pd
import matplotlib.pyplot as plt
import folium

def process_and_visualize_data():
    # Load and clean data
    #json_data = pd.read_json('meteorite_data.json')
    data = pd.read_csv('meteorite_data.csv')
    #data = pd.concat([json_data, csv_data], ignore_index=True)
    data = data.drop_duplicates(subset=['name', 'year', 'reclat', 'reclong'])
    data = data.dropna()
    
    # Trend Analysis
    data['year'] = data['year'].astype(int)
    data1 = data[data['year'] > 1800]
    landings_by_year = data1.groupby('year')['name'].count()
    
    # Plot
    # Highlight a year with a significant number of landings
    max_year = landings_by_year.idxmax()
    max_value = landings_by_year.max()

    plt.style.use('seaborn-v0_8-dark')

    plt.figure(figsize=(12, 8))
    plt.plot(
        landings_by_year.index, 
        landings_by_year.values, 
        color='blue', 
        marker='o', 
        linestyle='-', 
        linewidth=2, 
        markersize=5, 
        label='Meteorite Landings Over Time'
    )
    #plt.plot(landings_by_year.index, landings_by_year.values, label='Landings Over Time')
    plt.xlabel('Year', fontsize=16, labelpad=10)
    plt.ylabel('Number of Landings', fontsize=16, labelpad=10)
    plt.title('Meteorite Landings Over Time', fontsize=20, fontweight='bold', color='darkblue')
    plt.legend()
    plt.grid()
    plt.xlim(1850, 2008)
    plt.annotate(
        f'Highest: {max_value} landings',
        xy=(max_year, max_value),
        xytext=(max_year + 10, max_value - 50),
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=12,
        color='darkred'
    )
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.savefig('landings_by_year.png')

    # Map
    m = folium.Map(location=[0, 0], zoom_start=2)
    for _, row in data.iterrows():
        folium.CircleMarker(location=[row['reclat'], row['reclong']], radius=5).add_to(m)
    m.save('meteorite_map.html')

if __name__ == "__main__":
    process_and_visualize_data()
    print(plt.style.available)
