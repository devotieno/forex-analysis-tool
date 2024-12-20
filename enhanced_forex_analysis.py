import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict

class ForexAnalyzer:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.settings: Dict = {
            'sma_short': 20,
            'sma_long': 50,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2
        }

    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load and validate forex data.
        
        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        self.data = data.copy()
        
    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate technical indicators for forex analysis."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Moving Averages
        self.data['SMA_20'] = self.data['close'].rolling(
            window=self.settings['sma_short']).mean()
        self.data['SMA_50'] = self.data['close'].rolling(
            window=self.settings['sma_long']).mean()
        
        # RSI
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.settings['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.settings['rsi_period']).mean()
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['close'].ewm(
            span=self.settings['macd_fast'], adjust=False).mean()
        exp2 = self.data['close'].ewm(
            span=self.settings['macd_slow'], adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(
            span=self.settings['macd_signal'], adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']
        
        # Bollinger Bands
        self.data['BB_middle'] = self.data['close'].rolling(
            window=self.settings['bb_period']).mean()
        bb_std = self.data['close'].rolling(window=self.settings['bb_period']).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * self.settings['bb_std'])
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * self.settings['bb_std'])
        
        # ATR (Average True Range)
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.data['ATR'] = true_range.rolling(window=14).mean()
        
        return self.data

    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals based on multiple indicators."""
        self.data['Signal'] = 0
        
        # Price crosses SMA
        self.data['SMA_Cross'] = np.where(
            self.data['close'] > self.data['SMA_20'], 1,
            np.where(self.data['close'] < self.data['SMA_20'], -1, 0)
        )
        
        # RSI signals
        self.data['RSI_Signal'] = np.where(
            self.data['RSI'] < 30, 1,
            np.where(self.data['RSI'] > 70, -1, 0)
        )
        
        # MACD signals
        self.data['MACD_Signal'] = np.where(
            self.data['MACD'] > self.data['Signal_Line'], 1,
            np.where(self.data['MACD'] < self.data['Signal_Line'], -1, 0)
        )
        
        # Bollinger Bands signals
        self.data['BB_Signal'] = np.where(
            self.data['close'] < self.data['BB_lower'], 1,
            np.where(self.data['close'] > self.data['BB_upper'], -1, 0)
        )
        
        # Combined signal (weighted average)
        self.data['Combined_Signal'] = (
            self.data['SMA_Cross'] * 0.3 +
            self.data['RSI_Signal'] * 0.2 +
            self.data['MACD_Signal'] * 0.3 +
            self.data['BB_Signal'] * 0.2
        )
        
        return self.data

    def analyze_volatility(self) -> Dict[str, float]:
        """Calculate various volatility metrics."""
        volatility_metrics = {
            'std_dev': self.data['close'].std(),
            'avg_atr': self.data['ATR'].mean(),
            'max_drawdown': self._calculate_max_drawdown(),
            'daily_range': (
                (self.data['high'] - self.data['low']) / self.data['low']
            ).mean() * 100
        }
        return volatility_metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        roll_max = self.data['close'].expanding().max()
        daily_drawdown = self.data['close'] / roll_max - 1.0
        return daily_drawdown.min() * 100

def get_forex_data(symbol: str, period: str = "1mo", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch forex data from Yahoo Finance with error handling and validation.
    
    Args:
        symbol (str): Forex pair symbol (e.g., "EURUSD=X")
        period (str): Time period for data
        interval (str): Data interval
    
    Returns:
        pd.DataFrame: Processed forex data
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
            
        # Standardize column names
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add datetime index as column
        data = data.reset_index()
        data = data.rename(columns={'Date': 'datetime'})
        
        # Handle missing values
        data = data.fillna(method='ffill')
        
        return data
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def plot_analysis(analyzed_data: pd.DataFrame, signals: pd.DataFrame) -> None:
    """Create comprehensive visualization of the analysis."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))
    
    # Plot 1: Price Action with Indicators
    ax1.plot(analyzed_data.index, analyzed_data['close'], label='Price', alpha=0.7)
    ax1.plot(analyzed_data.index, analyzed_data['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(analyzed_data.index, analyzed_data['SMA_50'], label='SMA 50', alpha=0.7)
    ax1.fill_between(analyzed_data.index, 
                     analyzed_data['BB_upper'], 
                     analyzed_data['BB_lower'], 
                     alpha=0.1,
                     label='Bollinger Bands')
    ax1.set_title('Price Action with Indicators')
    ax1.legend()
    
    # Plot 2: RSI
    ax2.plot(analyzed_data.index, analyzed_data['RSI'], label='RSI')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.set_title('Relative Strength Index')
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    # Plot 3: MACD
    ax3.plot(analyzed_data.index, analyzed_data['MACD'], label='MACD')
    ax3.plot(analyzed_data.index, analyzed_data['Signal_Line'], label='Signal Line')
    ax3.bar(analyzed_data.index, analyzed_data['MACD_Histogram'], 
            label='MACD Histogram', alpha=0.3)
    ax3.set_title('MACD')
    ax3.legend()
    
    # Plot 4: Combined Signals
    colors = ['red' if x < 0 else 'green' for x in signals['Combined_Signal']]
    ax4.bar(signals.index, signals['Combined_Signal'], 
            label='Combined Signal', color=colors)
    ax4.set_title('Trading Signals')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def run_analysis(symbol: str = "EURUSD=X", 
                period: str = "1mo", 
                interval: str = "1h") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Run complete forex analysis pipeline with enhanced reporting.
    
    Args:
        symbol (str): Forex pair symbol
        period (str): Analysis period
        interval (str): Data interval
    
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: Analyzed data and signals
    """
    print(f"Analyzing {symbol}...")
    data = get_forex_data(symbol, period, interval)
    
    if data.empty:
        print("Analysis failed: No data available")
        return None, None
        
    analyzer = ForexAnalyzer()
    analyzer.load_data(data)
    
    analyzed_data = analyzer.calculate_indicators()
    signals = analyzer.generate_signals()
    volatility = analyzer.analyze_volatility()
    
    # Print comprehensive analysis
    print("\n=== Analysis Summary ===")
    latest = analyzed_data.iloc[-1]
    
    print(f"\nPrice Analysis:")
    print(f"Current Price: {latest['close']:.4f}")
    print(f"SMA 20: {latest['SMA_20']:.4f}")
    print(f"SMA 50: {latest['SMA_50']:.4f}")
    
    print(f"\nTechnical Indicators:")
    print(f"RSI: {latest['RSI']:.2f}")
    print(f"MACD: {latest['MACD']:.6f}")
    print(f"MACD Signal: {latest['Signal_Line']:.6f}")
    
    print(f"\nVolatility Metrics:")
    print(f"Standard Deviation: {volatility['std_dev']:.4f}")
    print(f"Average ATR: {volatility['avg_atr']:.4f}")
    print(f"Maximum Drawdown: {volatility['max_drawdown']:.2f}%")
    print(f"Average Daily Range: {volatility['daily_range']:.2f}%")
    
    print(f"\nTrading Signals:")
    signal_strength = latest['Combined_Signal']
    print(f"Signal Strength: {signal_strength:.2f}")
    
    if signal_strength > 0.5:
        print("Strong Buy Signal")
    elif signal_strength > 0:
        print("Weak Buy Signal")
    elif signal_strength < -0.5:
        print("Strong Sell Signal")
    elif signal_strength < 0:
        print("Weak Sell Signal")
    else:
        print("Neutral Signal")
    
    # Plot the analysis
    plot_analysis(analyzed_data, signals)
    
    return analyzed_data, signals

if __name__ == "__main__":
    # Example usage
    analyzed_data, signals = run_analysis(
        symbol="EURUSD=X",
        period="1mo",
        interval="1h"
    )
    # cd "Desktop\trading tool"
    # pip install yfinance pandas matplotlib