import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class StockPredictor:
    def __init__(self, ticker="AAPL", days_back=365, prediction_days=30):
        self.ticker = ticker
        self.days_back = days_back
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            return data
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def prepare_data(self, data, lookback=20):
        """Prepare features and labels for training"""
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(lookback, len(scaled_prices)):
            X.append(scaled_prices[i-lookback:i, 0])
            y.append(scaled_prices[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(self, data):
        """Train the prediction model"""
        X, y = self.prepare_data(data)
        self.model.fit(X, y)
    
    def predict_future(self, data, days=30):
        """Predict future stock prices"""
        last_prices = data['Close'].values[-20:].reshape(-1, 1)
        scaled = self.scaler.transform(last_prices)
        current_seq = scaled.flatten()
        
        predictions = []
        for _ in range(days):
            next_pred = self.model.predict([current_seq])[0]
            predictions.append(next_pred)
            current_seq = np.append(current_seq[1:], next_pred)
        
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()

class StockPredictorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor")
        self.root.geometry("1400x700")
        self.root.configure(bg='#f0f0f0')
        
        # Top 10 largest companies by market cap
        self.top_stocks = {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Alphabet": "GOOGL",
            "Amazon": "AMZN",
            "Tesla": "TSLA",
            "Berkshire Hathaway": "BRK.B",
            "Meta": "META",
            "NVIDIA": "NVDA",
            "JPMorgan Chase": "JPM",
            "Visa": "V"
        }
        
        self.color_options = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                             "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        
        self.predictions = {}
        self.historical_data = {}
        
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Control Panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)
        
        # Title
        title_label = ttk.Label(control_frame, text="Stock Predictor", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Stock Selection
        ttk.Label(control_frame, text="Select Stocks:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.stock_listbox = tk.Listbox(control_frame, height=10, width=20, selectmode=tk.MULTIPLE)
        for stock_name in self.top_stocks.keys():
            self.stock_listbox.insert(tk.END, stock_name)
        self.stock_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Timeline Selection
        ttk.Label(control_frame, text="Historical Data:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.timeline_var = tk.StringVar(value="365")
        for days, label in [("90", "3 months"), ("180", "6 months"), ("365", "1 year"), ("730", "2 years")]:
            ttk.Radiobutton(control_frame, text=label, variable=self.timeline_var, value=days).pack(anchor=tk.W)
        
        # Prediction Days
        ttk.Label(control_frame, text="Prediction Days:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.pred_days_var = tk.StringVar(value="30")
        for days in ["7", "14", "30", "60"]:
            ttk.Radiobutton(control_frame, text=f"{days} days", variable=self.pred_days_var, value=days).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="Generate Chart", command=self.generate_predictions).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="blue")
        self.status_label.pack(anchor=tk.W, pady=10)
        
        # Chart Area
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def generate_predictions(self):
        """Generate predictions for selected stocks"""
        selected_indices = self.stock_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one stock")
            return
        
        selected_stocks = [list(self.top_stocks.keys())[i] for i in selected_indices]
        days_back = int(self.timeline_var.get())
        pred_days = int(self.pred_days_var.get())
        
        self.predictions = {}
        self.historical_data = {}
        
        self.status_label.config(text="Loading data...", foreground="orange")
        self.root.update()
        
        # Fetch data and generate predictions
        for idx, stock_name in enumerate(selected_stocks):
            ticker = self.top_stocks[stock_name]
            try:
                predictor = StockPredictor(ticker=ticker, days_back=days_back, prediction_days=pred_days)
                historical_data = predictor.fetch_data()
                
                if len(historical_data) == 0:
                    messagebox.showerror("Error", f"No data found for {stock_name}")
                    return
                
                predictor.train(historical_data)
                future_predictions = predictor.predict_future(historical_data, days=pred_days)
                
                self.historical_data[stock_name] = historical_data
                self.predictions[stock_name] = future_predictions
                
                self.status_label.config(text=f"Loaded {idx+1}/{len(selected_stocks)}", foreground="blue")
                self.root.update()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to fetch data for {stock_name}: {str(e)}")
                self.status_label.config(text="Error loading data", foreground="red")
                return
        
        # Plot results
        self.plot_results(selected_stocks)
        self.status_label.config(text="Chart generated successfully", foreground="green")
    
    def plot_results(self, selected_stocks):
        """Plot historical and predicted prices"""
        # Clear previous plots
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        for idx, stock_name in enumerate(selected_stocks):
            historical_data = self.historical_data[stock_name]
            future_predictions = self.predictions[stock_name]
            
            historical_dates = historical_data.index
            future_dates = [historical_dates[-1] + timedelta(days=i+1) for i in range(len(future_predictions))]
            
            color = self.color_options[idx % len(self.color_options)]
            
            # Plot historical prices
            ax.plot(historical_dates, historical_data['Close'], linewidth=2, color=color, label=f'{stock_name} (History)')
            # Plot predicted prices
            ax.plot(future_dates, future_predictions, linewidth=2, linestyle='--', color=color, label=f'{stock_name} (Prediction)', alpha=0.7)
        
        ax.set_title('Stock Price Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def clear_all(self):
        """Clear all selections and plots"""
        self.stock_listbox.selection_clear(0, tk.END)
        self.predictions = {}
        self.historical_data = {}
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        self.status_label.config(text="Ready", foreground="blue")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorUI(root)
    root.mainloop()