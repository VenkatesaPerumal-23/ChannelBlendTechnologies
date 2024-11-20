**Collaborative Filtering Recommender System**
Overview
This project demonstrates the implementation of a collaborative filtering-based recommendation system using Surprise library's SVD algorithm. It predicts user preferences for products and recommends top-rated products based on their past interactions. Additionally, a simple API is deployed using Flask to serve recommendations dynamically.

Code Explanation
1. Import Libraries
pandas, numpy: For data manipulation and analysis.
matplotlib, seaborn: For visualizing product categories and ratings distribution.
surprise: For collaborative filtering using Singular Value Decomposition (SVD).
flask: To create a lightweight API for serving recommendations.
2. Dataset Creation
A mock dataset (mock_retail_data.csv) is created to simulate a retail environment with:
UserID: Represents unique users.
ProductID: Represents unique products.
ProductCategory: Product types like Electronics, Grocery, etc.
Rating: User ratings on a scale of 1-5.
The dataset is saved as a CSV file and loaded into a pandas DataFrame.

3. Exploratory Data Analysis (EDA)
Basic dataset statistics, data types, and value counts for product categories are displayed.
A bar chart visualizes the distribution of product categories, helping understand user behavior.
4. Preprocessing
The dataset is converted into the Surprise library's format using the Reader class. The rating_scale parameter specifies the range of ratings (1-5).
Data is split into training and test sets using train_test_split for model evaluation.
5. Collaborative Filtering Using SVD
The Singular Value Decomposition (SVD) algorithm is implemented to predict user-product interactions.
The model is trained using the training dataset (trainset), and performance is evaluated using RMSE (Root Mean Squared Error) on the test dataset (testset).
6. Recommendations Function
recommend_products():
Identifies products not yet rated by a user.
Predicts ratings for these products using the trained SVD model.
Returns the top-N products with the highest predicted ratings.
7. Flask API for Recommendations
A simple Flask-based API is created to serve recommendations dynamically:
Endpoint: /recommend
Method: POST
Input: JSON with a user ID (e.g., {"UserID": "U1"}).
Output: JSON with top product recommendations for the given user.
8. Example Outputs
Training Output:
Displays RMSE score, indicating model performance.

Recommendation Example:
Input: UserID = "U1"
Output: Top 5 recommended products (e.g., ["P4","P2",...]).

API Example:
Request:
{
    "UserID": "U1"
}

Response:
{
    "UserID": "U1",
    "Recommendations": ["P4","P2","P5" ...]
}


**How to Run the Code**
1.Install Python 3.8+.
2.pip install pandas numpy matplotlib seaborn flask scikit-surprise
3.Clone the repository:
  git clone https://github.com/VenkatesaPerumal-23/ChannelBlendTechnologies.git
4.Navigate to the project directory:
  cd Collaborative-Filtering-Recommender
Run the script:
  python main.py'
  
To start the Flask API:
python main.py
  The API will run on http://127.0.0.1:5000/.
Accessing the API
Use any API testing tool like Postman or cURL:
curl -X POST http://127.0.0.1:5000/recommend -H "Content-Type: application/json" -d '{"UserID": "U1"}'
