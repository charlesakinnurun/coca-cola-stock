# Coca-Cola Stock Price
![Coca-Cola Stock Price](/image.jpg)

## Overview
The goal is to predict the Closing Price for the Next Trading Day

## Procedures
- Data Acquistion
    - Use yfinance historical stock data
- Data Preprocessing
    - Handle missing values
    - Handle duplicated rows
- Pre-Training Visualization
    - Historical Close Price using Line plot
- Feature Engineering
- Data Splitting
    - 80% training and testing 20% 
    - shuffle=False
- Data Scaling
    - StandardScaler
    - Stanadardize features by removing the mean and scaling to unit variance
- Hyperparamter Grids
- Model Definitions
    - Linear Regression (Baseline)
    - Ridge Regression
    - Lasso Regression
    - Decison Tree Regression
    - Random Forest Regression
    - Support Vector Regression
- Model Training
- Hyperparameter Tuning
- Post-Training Visualization
- New prediction

### Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/coca-cola-stock.git
cd coca-cola-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Tools and Dependencies
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinace
- Environment
    - Jupyter Notebook
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Project Structure
```
coca-cola-stock/
│
├── model.ipynb  
|── model.py    
├── requirements.txt 
├── LICENSE
├── image.jpg    
└── README.md          
```
## Contributing
Contributions are welcome! If you’d like to suggest improvements — e.g., new modelling algorithms, additional feature engineering, or better documentation — please open an Issue or submit a Pull Request.
Please ensure your additions are accompanied by clear documentation and, where relevant, updated evaluation results.

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.