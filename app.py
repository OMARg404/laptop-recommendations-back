from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the React front-end

# Load the dataset (CSV file)
file_path = r"./mainfilter.csv"  # Update path if needed
df = pd.read_csv(file_path)  # Use pd.read_csv for .csv file

# Euclidean distance function
def euclidean_distance(selected_features, product_features):
    distance = 0
    for feature in selected_features:
        if pd.notnull(selected_features[feature]) and pd.notnull(product_features[feature]):
            distance += (selected_features[feature] - product_features[feature]) ** 2
    return np.sqrt(distance)

# Filtering function for the laptop data
def filter_laptops(df, brand=None, cpu=None, gpu=None, storage_range=None, ram_range=None, screen_range=None, price_range=None):
    filtered = df.copy()

    # Ensure correct data types for filtering
    filtered['RAM'] = pd.to_numeric(filtered['RAM'], errors='coerce')
    filtered['Storage'] = pd.to_numeric(filtered['Storage'], errors='coerce')
    filtered['Screen'] = pd.to_numeric(filtered['Screen'], errors='coerce')
    filtered['FinalPrice'] = pd.to_numeric(filtered['FinalPrice'], errors='coerce')

    # Apply filters
    if brand:
        filtered = filtered[filtered['Brand'].str.contains(brand, case=False, na=False)]
    if cpu:
        filtered = filtered[filtered['CPU'].str.contains(cpu, case=False, na=False)]
    if gpu:
        filtered = filtered[filtered['GPU'].str.contains(gpu, case=False, na=False)]

    if storage_range and len(storage_range) == 2:
        min_storage, max_storage = storage_range
        filtered = filtered[(filtered['Storage'] >= min_storage) & (filtered['Storage'] <= max_storage)]

    if ram_range and len(ram_range) == 2:
        min_ram, max_ram = ram_range
        filtered = filtered[(filtered['RAM'] >= min_ram) & (filtered['RAM'] <= max_ram)]

    if screen_range and len(screen_range) == 2:
        min_screen, max_screen = screen_range
        filtered = filtered[(filtered['Screen'] >= min_screen) & (filtered['Screen'] <= max_screen)]

    if price_range and len(price_range) == 2:
        min_price, max_price = price_range
        filtered = filtered[(filtered['FinalPrice'] >= min_price) & (filtered['FinalPrice'] <= max_price)]

    return filtered

@app.route('/recommend', methods=['POST'])
def recommend_laptops():
    preferences = request.get_json()

    brand = preferences.get('brand', None)
    cpu = preferences.get('cpu', None)
    gpu = preferences.get('gpu', None)
    storage_range = preferences.get('storage', None)
    ram_range = preferences.get('ram', None)
    screen_range = preferences.get('screen', None)
    price_range = preferences.get('price', None)

    # Ensure storage_range, ram_range, screen_range, and price_range are lists with two elements
    def validate_range(range_value):
        if range_value and isinstance(range_value, list) and len(range_value) == 2:
            return range_value
        return None

    storage_range = validate_range(storage_range)
    ram_range = validate_range(ram_range)
    screen_range = validate_range(screen_range)
    price_range = validate_range(price_range)

    filtered_laptops = filter_laptops(
        df,
        brand=brand,
        cpu=cpu,
        gpu=gpu,
        storage_range=storage_range,
        ram_range=ram_range,
        screen_range=screen_range,
        price_range=price_range
    )

    # Compute the Euclidean distance for each filtered laptop
    def compute_distance(product):
        selected_features = {
            'RAM': preferences.get('ram', [0, 64])[0],  # Take the lower bound of RAM range as the preference
            'Storage': preferences.get('storage', [0, 1000])[0],  # Similarly for other features
            'Screen': preferences.get('screen', [0, 20])[0],
            'FinalPrice': preferences.get('price', [0, 1900])[0]
        }
        return euclidean_distance(selected_features, product)

    filtered_laptops['distance'] = filtered_laptops.apply(compute_distance, axis=1)
    filtered_laptops = filtered_laptops.sort_values(by='distance')

    # Return the top 20 nearest products
    recommended_laptops = filtered_laptops.head(20).to_dict(orient="records")

    return jsonify(recommended_laptops)

if __name__ == '__main__':
    app.run(debug=True)
