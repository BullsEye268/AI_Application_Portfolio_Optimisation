import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.covariance import LedoitWolf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
import pymongo
from pathlib import Path
import mlflow
import mlflow.sklearn
import logging
import time
import itertools
from tqdm import tqdm

# Load environment variables
load_dotenv()

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(log_dir, "web_application.log")
logging.basicConfig(
    level=logging.WARN,                      # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    filename=log_file,              # Log file name
    filemode='a',                            # Append mode ('w' would overwrite)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'             # Date format
)
logging.Formatter.converter = time.localtime
logger = logging.getLogger(__name__)


# Define the stock symbols - these are just examples, can be changed
STOCK_SYMBOLS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'V']
DAYS = 300

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


def fetch_stock_data():
    """Fetch historical stock data and save it using DVC."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS)
    
    all_data = pd.DataFrame()
    
    for symbol in STOCK_SYMBOLS:
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if not data.empty:
                # Calculate daily returns
                data['Returns'] = data['Close'].pct_change()
                # Add symbol column
                data['Symbol'] = symbol
                # Append to main dataframe
                all_data = pd.concat([all_data, data])
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    # Save data
    all_data.to_csv('data/stock_data.csv')
    return all_data


def prepare_training_data(all_data):
    """Prepare data for training the portfolio optimization model."""
    # Reshape data for returns calculation
    returns_data = all_data.reset_index().pivot(index='Date', columns='Symbol', values='Returns')
    returns_data = returns_data.dropna()
    
    return returns_data


def evaluate_portfolio(weights, returns_data):
    """Evaluate portfolio performance for a set of weights."""
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Calculate portfolio performance
    portfolio_return = np.sum(returns_data.mean() * weights) * 252  # Annualized return
    
    # Use Ledoit-Wolf covariance estimator (robust to estimation error)
    cov_estimator = LedoitWolf().fit(returns_data)
    cov_matrix = cov_estimator.covariance_
    
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return {
        'weights': weights,
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }


def train_model():
    """Train the portfolio optimization model using grid search."""
    # Fetch and prepare data
    logger.info('Starting Training')
    all_data = fetch_stock_data()
    logger.info('Processing Raw Data')
    print(all_data)
    returns_data = prepare_training_data(all_data)
    
    # Save prepared data for DVC tracking
    returns_data.to_csv('data/returns_data.csv')
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("portfolio_optimization")
    
    with mlflow.start_run(run_name="grid_search_optimization"):
        # Define a grid of weight allocations
        # We'll use a coarse grid to keep computation manageable
        # Each stock can have weights of 0.0, 0.2, 0.4, 0.6, 0.8, or 1.0
        # but we'll filter to only keep combinations where weights sum approximately to 1
        
        # For a 10-stock portfolio, a full grid would be too large
        # Let's start with fixed steps and filter combinations
        weights_grid = []
        
        # Define possible weight values for each stock
        weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Generate initial weight combinations for first 5 stocks
        # We'll set constraints for the remaining stocks based on these
        for combo in itertools.product(weight_values, repeat=5):
            # If the sum is already > 1, skip this combination
            if sum(combo) > 1.0:
                continue
                
            # Calculate remaining weight for other stocks
            remaining_weight = 1.0 - sum(combo)
            
            # Set a simple allocation for the remaining stocks
            # Distribute the remaining weight equally among the other stocks
            remaining_per_stock = remaining_weight / 5
            
            # Create the full weight vector
            weights = list(combo) + [remaining_per_stock] * 5
            
            # Add to our grid
            weights_grid.append(weights)
        
        # Add more specific weight combinations
        # Equal weights
        weights_grid.append([0.1] * 10)
        
        # Tech-heavy portfolio
        tech_weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
        weights_grid.append(tech_weights)
        
        # Finance-heavy portfolio
        finance_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.3, 0.3]
        weights_grid.append(finance_weights)
        
        # Mixed portfolio
        mixed_weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
        weights_grid.append(mixed_weights)
        
        print(f"Evaluating {len(weights_grid)} portfolio configurations...")
        
        # Evaluate each combination
        best_sharpe_ratio = -float('inf')
        best_portfolio = None
        
        for weights in tqdm(weights_grid):
            weights = np.array(weights)
            
            # Evaluate portfolio
            portfolio = evaluate_portfolio(weights, returns_data)
            
            # Log to MLflow
            with mlflow.start_run(run_name="portfolio_config", nested=True):
                # Log parameters (weights)
                for i, symbol in enumerate(STOCK_SYMBOLS):
                    mlflow.log_param(f"weight_{symbol}", portfolio['weights'][i])
                
                # Log metrics
                mlflow.log_metric("sharpe_ratio", portfolio['sharpe_ratio'])
                mlflow.log_metric("portfolio_return", portfolio['portfolio_return'])
                mlflow.log_metric("portfolio_volatility", portfolio['portfolio_volatility'])
            
            # Track best portfolio
            if portfolio['sharpe_ratio'] > best_sharpe_ratio:
                best_sharpe_ratio = portfolio['sharpe_ratio']
                best_portfolio = portfolio
        
        # Extract the best weights
        best_weights = best_portfolio['weights']
        
        # Create model dictionary
        model = {
            'weights': best_weights.tolist(),
            'symbols': STOCK_SYMBOLS,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'metrics': {
                'sharpe_ratio': best_sharpe_ratio,
                'portfolio_return': best_portfolio['portfolio_return'],
                'portfolio_volatility': best_portfolio['portfolio_volatility']
            }
        }
        
        # Save model with MLflow
        mlflow.log_dict(model, "model/portfolio_model.json")
        
        # Save model locally for compatibility with existing code
        joblib.dump(model, 'models/portfolio_model.joblib')
        
        # Save metrics for DVC tracking
        with open('results/metrics.json', 'w') as f:
            json.dump({
                'sharpe_ratio': float(best_sharpe_ratio),
                'portfolio_return': float(best_portfolio['portfolio_return']),
                'portfolio_volatility': float(best_portfolio['portfolio_volatility']),
                'training_date': datetime.now().strftime('%Y-%m-%d')
            }, f)
        
        # Create visualization of optimal portfolio allocation
        plt.figure(figsize=(12, 8))
        plt.bar(STOCK_SYMBOLS, best_weights)
        plt.xlabel('Stocks')
        plt.ylabel('Weight')
        plt.title('Optimal Portfolio Allocation')
        plt.savefig('results/portfolio_allocation.png')

        # Log figure with MLflow
        mlflow.log_artifact('results/portfolio_allocation.png')
        
        # Store model in MongoDB if connection details are provided
        print('trying to save model to mongodb')
        mongo_uri = os.getenv('MONGO_URI')
        if mongo_uri:
            try:
                client = pymongo.MongoClient(mongo_uri)
                db = client.portfolio_optimization
                collection = db.models
                
                # Convert numpy arrays to lists for MongoDB
                model_doc = {
                    'weights': best_weights.tolist(),
                    'symbols': STOCK_SYMBOLS,
                    'training_date': datetime.now(),
                    'sharpe_ratio': float(best_sharpe_ratio),
                    'portfolio_return': float(best_portfolio['portfolio_return']),
                    'portfolio_volatility': float(best_portfolio['portfolio_volatility']),
                    'mlflow_run_id': mlflow.active_run().info.run_id
                }
                
                collection.insert_one(model_doc)
                print("Model saved to MongoDB")
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")
        
        return model


if __name__ == "__main__":
    train_model() 