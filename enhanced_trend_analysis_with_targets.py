import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EnhancedTrendAnalyzer:
    def __init__(self, symbol, short_window=12, long_window=26, rsi_period=14, target_pct=0.02):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.target_pct = target_pct  # Target price percentage
        self.data = None

    def load_data(self, period="1y", interval="1d"):
        """Load historical data from Yahoo Finance."""
        try:
            self.data = yf.download(self.symbol, period=period, interval=interval, timeout=30)
            if self.data.empty:
                print("No data returned. Please check the symbol and parameters.")
                return False
            self.data.reset_index(inplace=True)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def calculate_indicators(self):
        """Calculate EMAs and RSI."""
        # Calculate EMAs
        self.data['EMA_Short'] = self.data['Close'].ewm(span=self.short_window, adjust=False).mean()
        self.data['EMA_Long'] = self.data['Close'].ewm(span=self.long_window, adjust=False).mean()

        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

    def generate_signals(self):
        """Generate buy/sell signals based on EMA crossovers and RSI, and calculate price targets."""
        self.data['Signal'] = 0
        self.data['Target'] = np.nan  # Initialize target column

        # Generate signals
        self.data['Signal'] = np.where(
            (self.data['EMA_Short'] > self.data['EMA_Long']) & (self.data['RSI'] < 70), 1, 0
        )
        self.data['Signal'] = np.where(
            (self.data['EMA_Short'] < self.data['EMA_Long']) & (self.data['RSI'] > 30), -1, self.data['Signal']
        )
        
        # Calculate position
        self.data['Position'] = self.data['Signal'].diff()

        # Calculate targets based on signals
        for i in range(1, len(self.data)):
            if self.data['Position'].iloc[i] == 1:  # Buy signal
                self.data['Target'].iloc[i] = self.data['Close'].iloc[i] * (1 + self.target_pct)
            elif self.data['Position'].iloc[i] == -1:  # Sell signal
                self.data['Target'].iloc[i] = self.data['Close'].iloc[i] * (1 - self.target_pct)

    def plot_results(self):
        """Plot the closing price, EMAs, RSI, and targets."""
        plt.figure(figsize=(14, 12))

        # Price and EMAs
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(self.data['Date'], self.data['Close'], label='Close Price', alpha=0.5)
        ax1.plot(self.data['Date'], self.data['EMA_Short'], label='Short-Term EMA', alpha=0.75)
        ax1.plot(self.data['Date'], self.data['EMA_Long'], label='Long-Term EMA', alpha=0.75)

        # Plot buy signals
        ax1.plot(self.data[self.data['Position'] == 1]['Date'],
                 self.data[self.data['Position'] == 1]['Close'],
                 '^', markersize=10, color='g', lw=0, label='Buy Signal')

        # Plot sell signals
        ax1.plot(self.data[self.data['Position'] == -1]['Date'],
                 self.data[self.data['Position'] == -1]['Close'],
                 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        # Plot targets
        ax1.plot(self.data['Date'], self.data['Target'], 'o', markersize=5, color='orange', label='Target Price')

        ax1.set_title(f'Trend Analysis for {self.symbol}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid()

        # RSI Plot
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(self.data['Date'], self.data['RSI'], label='RSI', color='b')
        ax2.axhline(70, linestyle='--', alpha=0.5, color='r')
        ax2.axhline(30, linestyle='--', alpha=0.5, color='g')
        ax2.set_title('Relative Strength Index')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        if self.load_data():
            self.calculate_indicators()
            self.generate_signals()
            self.plot_results()
        else:
            print("Data loading failed. Analysis cannot be performed.")


if __name__ == "__main__":
    # Example usage
    symbol = "EURUSD=X"  # Change to any trading symbol as needed
    analyzer = EnhancedTrendAnalyzer(symbol)
    analyzer.run_analysis()