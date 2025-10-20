# %% [markdown]
# Import the neccessary libraries

# %%
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

# %% [markdown]
# Set plotting style for better visualization

# %%
plt.style.use("seaborn-v0_8-whitegrid")

# %% [markdown]
# Configuration and Constants

# %%
TICKER = "KO" # Coca-cola company stock ticker
START_DATE = "2010-01-01" # Starting date for data visualization
END_DATE = pd.to_datetime("today").strftime('%Y-%m-%d') # Today's date as end date
PREDICTION_DAYS = 1 # We aim to predict the stock price 1 day into a future

# %% [markdown]
# Data Loading and Acquistion

# %%
print(f"Fetching {TICKER} stock data from {START_DATE} to {END_DATE}")

# Use yfinane to download historical stock data
try:
    df = yf.download(TICKER,start=START_DATE,end=END_DATE)
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# %%
# Check if data was successfully loaded and is not empty
if df.empty:
    print(f"No data retrived for {TICKER}. Exiting...")
    exit()

# %%
df.reset_index(inplace=True)

# %%
# Display the first few rows of the raw data
print("----- Raw Head -----")
df.head()

# %%
# Display the data information
print(df.info())

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %% [markdown]
# Visualization Before Training

# %%
# Display a simple plot showing the historical close price (before training)
plt.Figure(figsize=(14,3))
df["Close"].plot(title=f"Historical {TICKER} Close Price (Data Before Training)",color="green")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Feature Engineering

# %%
# Create features based on the current day's trading data
features = df[["Open","High","Low","Close","Volume"]].copy()

# %%
# 1. Target Variable Creation: We want to predict the "Close" price for the next day
# We shift the "Close" column upward to -PREDICTION_DAYS (1 day) to align the future value
# (Target: y) with current day's features (Input: X)
features["Target"] = features["Close"].shift(-PREDICTION_DAYS)

# %%
# 2. Feature Engineering: Adding simple moving averages (SMAs) as technical indicators
# SMAs smooth out price data to identify trend direction
features["SMA_5"] = features["Close"].rolling(window=5).mean()
features["SMA_20"] = features["Close"].rolling(window=20).mean()

# %%
# Drop rows with NaN values created by shiftig and rolling window operators
# These NaNs are from the last row (Target) and the first 20 rows (SMA_20)
features.dropna(inplace=True)

# %%
# Define the feature matrix (X) and the targte Variable (y)
X =  features.drop("Target",axis=1) # All columns except "Target" are features
y = features["Target"]

# %%
# Store the dates corresponding to the data points for visualization
dates = features.index.values

# %% [markdown]
# Data Splitting

# %%
# Split the data into training (80%) and testing (20%) sets
# We don't shuffle time series data as the order is cruical
# We use a fixed split point based on the index (time)
split_point = int(0.8  * len(X))
X_train,X_test = X[:split_point],X[split_point:]
y_train,y_test = y[:split_point],y[split_point:]
dates_test = dates[split_point:] # Keep track of the test dates for plotting

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# %% [markdown]
# Data Scaling

# %%
# Stanadardize features by removing the mean and scaling to unit variance
# This is crucial for distance-based algorithms like SVR and Linear Models with regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test) # Use the fitted scaler from the training set

# %%
# Convert back to DataFrame for better visualization/debugging (optional, but good practice)
X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns)

# %% [markdown]
# Hyperparameter Grids

# %%
# 1. Linear Models (Ridge/Lasso/ElasticNet): Use 'alpha' (regularization strength)
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0]}

# 2. Decision Tree: Use 'max_depth' and 'min_samples_split'
dt_params = {
    'max_depth': np.arange(3, 15), # Max depth of the tree
    'min_samples_split': np.arange(2, 10) # Minimum number of samples required to split an internal node
}

# 3. Random Forest: Use 'n_estimators' (number of trees) and 'max_depth'
rf_params = {
    'n_estimators': np.arange(50, 200, 50), # Number of trees in the forest
    'max_depth': np.arange(5, 20),
    'min_samples_leaf': [1, 2, 4]
}

# 4. SVR: Use 'C' (regularization parameter) and 'gamma' (kernel coefficient)
svr_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'], # How much influence a single training example has
    'kernel': ['rbf']
}

# %% [markdown]
# Model Definitions

# %%
models_to_run = [
    ("Linear Regression (Baseline)",LinearRegression(),{}), # No tuning for simple Linear Regression
    ("Ridge Regression",Ridge(),ridge_params),
    ("Lasso Regression",Lasso(max_iter=5000),lasso_params),
    ("Decision Tree",DecisionTreeRegressor(random_state=42),dt_params),
    ("Random Forest",RandomForestRegressor(random_state=42),rf_params),
    ("SVR",SVR(),svr_params)
]

# %%
# Dictionary to store results for comparison
results = {}
best_model = None
best_score = -np.inf # Initialize with negative infinity for R-squared metric (higher is better)

# Define which models require feature scaling (all linear and distance-based models)
SCALED_MODELS = ["Linear Regression (Baseline)","Ridge Regression","Lasso Regression","SVR"]

# %% [markdown]
# Model Training and Hyperparameter Tuning

# %%
for name, model, params in models_to_run:
    print(f"\nTraining and Tuning: {name}...")

    # Determine the correct training and testing data based on whether the model requires scaling
    requires_scaling = name in SCALED_MODELS

    if requires_scaling:
        X_train_data = X_train_scaled
        X_test_pred = X_test_scaled
    else:
        # Tree-based models (DT, RF) do not benefit from scaling and can use the original data
        X_train_data = X_train
        X_test_pred = X_test

    # If parameters exist, use RandomizedSearchCV for tuning
    if params:
        # RandomizedSearchCV samples a fixed number of parameter settings from the search space.
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=10 if name != "SVR" else 5, # Run 10 iterations (fewer for slow SVR)
            scoring='r2',
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1 # Use all available CPU cores
        )
        
        # Fit the search object to the determined training data
        search.fit(X_train_data, y_train)

        # The best estimator is the model trained with the best hyperparameters
        final_model = search.best_estimator_
        best_params = search.best_params_
        print(f"   Best Params: {best_params}")
    else:
        # For models without tuning (like Linear Regression Baseline), just fit the model directly
        final_model = model
        final_model.fit(X_train_data, y_train)
        best_params = "N/A"

    # Make predictions on the test set using the correctly prepared test data
    y_pred = final_model.predict(X_test_pred)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results[name] = {'R2 Score': r2, 'MSE': mse, 'Model': final_model, 'Params': best_params, 'Predictions': y_pred}

    print(f"   Test R² Score: {r2:.4f}")
    print(f"   Test MSE: {mse:.4f}")

    # Check for the best model based on R-squared (coefficient of determination)
    if r2 > best_score:
        best_score = r2
        best_model_name = name
        best_model = final_model

# --- Final Comparison and Best Model Selection ---
print("\n--- Final Model Comparison ---")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R2 Score': [r['R2 Score'] for r in results.values()],
    'MSE': [r['MSE'] for r in results.values()],
    'Best Params': [r['Params'] for r in results.values()]
}).sort_values(by='R2 Score', ascending=False).set_index('Model')
print(comparison_df.round(4))

print(f"\n*** The best performing model based on R² score is: {best_model_name} ***")
best_results = results[best_model_name]


# %% [markdown]
# Visualiztion: Actual vs Predicted (Best Model)

# %%
print("Generating Visualization..........")

plt.Figure(figsize=(14,6))

# Plot the actual (true) prices from the test set
plt.plot(dates_test,y_test.values,label="Actual Price",color="darkblue",linewidth=2)

# Plot the predicted prices from the best model
plt.plot(dates_test,best_results["Predictions"],label=f"Predicted Price ({best_model_name})",color="red",linestyle="--",alpha=0.7)
plt.title(f"{TICKER} Stock Price Prediction - Actual vs Predicted ({best_model_name})")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# New Prediction

# %%
def predict_next_day_price(model,latest_data):
    print("----- New Prediction Input (Next Trading Day)")


    # The latest data needs to be structured exactly like X_train (Open, High, Low, Close, Volume, SMA_5, SMA_20)
    # The last row of our preprocessed features DataFrame contains the data needed for the prediction.
    last_row = latest_data.iloc[-1].drop("Target")

    # Since the Random Forest was chosen as the best model, we do not need to scale it
    # If the best model was LinearRegression/Lasso/Ridge/SVR,scaling would be mandatory

    # 1. Reshape the data for prediction (single sample)
    X_new = last_row.values.reshape(1,-1)

    # 2. Check if scaling is needed for the best model
    # Note: SCALED_MODELS is a global constant defined in the main script now
    is_scaled = best_model_name in SCALED_MODELS


    if is_scaled:
        # Only scale the features if the best model requires scaled input (i-e., not a tree-based model)
        X_new_scaled = scaler.transform(X_new)
        prediction = model.predict(X_new_scaled)
    else:
        prediction = model.predict(X_new)

    print(f"Latest input features used for prediction (today's data):")
    print(last_row.to_frame().T)

    print(f"Predicted {TICKER} Closing Price fo the Next Trading Day:")
    print(f"${prediction[0]:.2f} (Using {best_model_name})")


    return prediction[0]




# Execute the prediction function using the best model and the last row of the original features

predict_next_day_price(best_model,features)


