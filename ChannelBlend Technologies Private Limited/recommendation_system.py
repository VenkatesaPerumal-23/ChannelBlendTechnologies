# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Mock Dataset Creation
def create_mock_dataset():
    data = {
        "UserID": ["U1", "U2", "U3", "U1", "U2", "U4", "U5", "U3", "U5", "U1"],
        "ProductID": ["P1", "P2", "P3", "P4", "P1", "P5", "P1", "P4", "P2", "P3"],
        "ProductCategory": ["Electronics", "Apparel", "Grocery", "Electronics", 
                            "Apparel", "Grocery", "Electronics", "Apparel", "Electronics", "Grocery"],
        "Rating": [5, 4, 3, 5, 4, 3, 5, 4, 3, 4],
    }
    df = pd.DataFrame(data)
    df.to_csv("mock_retail_data.csv", index=False)
    return df

# Load Dataset
df = create_mock_dataset()
print("Dataset Preview:")
print(df.head())

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    print("\nDataset Info:")
    print(data.info())

    print("\nSummary Statistics:")
    print(data.describe())

    print("\nValue Counts for Product Categories:")
    print(data['ProductCategory'].value_counts())

    plt.figure(figsize=(10, 6))
    sns.countplot(x='ProductCategory', data=data, palette='viridis')
    plt.title('Product Category Distribution')
    plt.show()

perform_eda(df)

# Preprocessing for Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserID', 'ProductID', 'Rating']], reader)

# Train-Test Split
trainset, testset = train_test_split(data, test_size=0.2)

# Model Development - Collaborative Filtering with SVD
print("\nTraining the model...")
model = SVD()
model.fit(trainset)

# Evaluate Model
print("\nEvaluating the model...")
predictions = model.test(testset)
rmse(predictions)

# Recommend Products Function
def recommend_products(user_id, model, data, n=5):
    unique_products = df['ProductID'].unique()
    user_rated_products = df[df['UserID'] == user_id]['ProductID'].unique()
    products_to_predict = [p for p in unique_products if p not in user_rated_products]

    predictions = [model.predict(user_id, product) for product in products_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_recommendations = predictions[:n]
    return [pred.iid for pred in top_recommendations]

# Test Recommendation System
user_to_test = "U1"
recommended_products = recommend_products(user_to_test, model, data)
print(f"\nTop Recommendations for User {user_to_test}: {recommended_products}")

# Optional: Deploy as a Simple API using Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['UserID']
    recommendations = recommend_products(user_id, model, data)
    return jsonify({"UserID": user_id, "Recommendations": recommendations})

if __name__ == '__main__':
    print("\nStarting Flask API...")
    app.run(debug=True)
