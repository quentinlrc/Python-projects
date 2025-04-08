import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import pytz
        
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    STRONG_BUY = 2
    MID_BUY = 1.5
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    MID_SELL = -1.5
    STRONG_SELL = -2

@dataclass
class WeightConfig:
    momentum: float = 0.45    # 45% weight 
    trend: float = 0.35       # 35% weight 
    volatility: float = 0.20  # 20% weight

@dataclass
class TradeSignal:
    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strength: float
    reasons: Dict[str, str]

# Path for pairs - Update these to your actual file paths
path_NZDUSD ='/Users/quentinleriche/Desktop/FX_DATA/NZDUSD15.csv'
path_USDCAD ='/Users/quentinleriche/Desktop/FX_DATA/USDCAD15.csv'
path_AUDUSD ='/Users/quentinleriche/Desktop/FX_DATA/AUDUSD15.csv'
path_USDCHF ='/Users/quentinleriche/Desktop/FX_DATA/USDCHF15.csv'
path_EURUSD ='/Users/quentinleriche/Desktop/FX_DATA/EURUSD15.csv'
path_GBPUSD ='/Users/quentinleriche/Desktop/FX_DATA/GBPUSD15.csv'

def get_forex_data(pair: str, start: datetime, end: datetime, interval: str = '15m') -> Optional[pd.DataFrame]:
    """
    Get forex data from local CSV files
    
    Args:
        pair: Currency pair (e.g., 'AUDUSD')
        start: Start date for full data load
        end: End date for full data load
        interval: Data interval (not used for CSV files, but kept for interface consistency)
        
    Returns:
        Pandas DataFrame with OHLCV data
    """
    try:
        # Select the appropriate path based on the pair
        if pair == 'NZDUSD':
            file_path = path_NZDUSD
        elif pair == 'USDCAD':
            file_path = path_USDCAD
        elif pair == 'AUDUSD':
            file_path = path_AUDUSD
        elif pair == 'USDCHF':
            file_path = path_USDCHF
        elif pair == 'EURUSD':
            file_path = path_EURUSD
        elif pair == 'GBPUSD':
            file_path = path_GBPUSD
        else:
            logger.error(f"No data file configured for pair {pair}")
            return None
        
        # Read the CSV file - try to handle different possible formats
        try:
            # First try with tab delimiter and no header
            data = pd.read_csv(file_path, delimiter='\t', header=None)
            if len(data.columns) < 5:  # Not enough columns
                raise ValueError("Not enough columns with tab delimiter")
            data.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
        except Exception as e:
            logger.warning(f"Failed with tab delimiter: {str(e)}, trying comma delimiter")
            try:
                # Try with comma delimiter and no header
                data = pd.read_csv(file_path, header=None)
                if len(data.columns) < 5:  # Not enough columns
                    raise ValueError("Not enough columns with comma delimiter")
                data.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
            except Exception as e2:
                logger.warning(f"Failed with comma delimiter and no header: {str(e2)}, trying with header")
                # Try with header
                data = pd.read_csv(file_path)
                # Verify essential columns exist
                required_cols = ["Time", "Open", "High", "Low", "Close"]
                if not all(col in data.columns for col in required_cols):
                    logger.error(f"CSV file for {pair} is missing required columns")
                    return None
        
        # Convert Time to datetime and set as index
        try:
            # Try different date formats
            data["Time"] = pd.to_datetime(data["Time"])
        except Exception as e:
            logger.warning(f"Failed to parse date: {str(e)}, trying different format")
            try:
                # Try another common format
                data["Time"] = pd.to_datetime(data["Time"], format="%Y.%m.%d %H:%M:%S")
            except Exception as e2:
                logger.error(f"Could not parse date column for {pair}: {str(e2)}")
                return None
        
        data.set_index("Time", inplace=True)
        
        
        # Filter data to the requested date range if possible
        data_start = data.index.min()
        data_end = data.index.max()
        
        if start > data_end or end < data_start:
            logger.error(f"Requested date range {start} to {end} outside available data range")
            return None
        
        # Find actual start date (use data_start if requested start is before available data)
        actual_start = max(start, data_start)
        actual_end = min(end, data_end)
        
        # Filter data 
        filtered_data = data[(data.index >= actual_start) & (data.index <= actual_end)]
        
        if filtered_data.empty:
            logger.error(f"No data available for {pair} in the specified date range")
            return None
        
        # Store backtest_start as an attribute for later use
        filtered_data.attrs['backtest_start'] = actual_start
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error loading data for {pair}: {str(e)}")
        return None
    
class ForexStrategy(bt.Strategy):
    params = {
        'ema_short': 18, # standard : 10
        'ema_long': 24, # standard : 20
        'rsi_period': 14, # standard : 14
        'atr_period': 14, # standard : 14
        'bb_period': 20, # standard : 20
        'stoch_period': 14, # standard : 14
        'macd_fast': 12, # standard : 12
        'macd_slow': 26, # standard : 26
        'macd_signal': 9,
        'atr_multiplier': 1.5,     # current best : 1.5
        'trailing_atr': True,      
        'trailing_percent': 0.5,   # current best : 0.6
        'risk_reward_ratio': 1.8,  # current best : 1.8
        'min_strength': 0.8,       # current best : 0.8
        'max_risk_percent': 1,   
        'lot_size': 10000          # standard mini lot size (10,000 units)
    }
    
    def __init__(self):
        # Initialize indicators
        self.ema_short = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.ema_short)
        self.ema_long = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.ema_long)
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close, period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(
            self.data, period=self.p.atr_period)
        self.bbands = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period)
        self.stoch = bt.indicators.Stochastic(
            self.data, period=self.p.stoch_period)
        self.macd = bt.indicators.MACD(
            self.data.close, 
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow, 
            period_signal=self.p.macd_signal)
        
        # Initialize weights
        self.weights = WeightConfig()
        
        # Strategy state variables
        self.open_position = None  # Track open position: None, 'BUY', or 'SELL'
        self.last_sl_hit = None    # Track direction of last stop loss hit
        self.last_exit_time = None # Track when the last position was closed
        
        # Trailing stop variables
        self.current_stop_loss = None  # Current stop loss level
        self.current_take_profit = None  # Current take profit level
        self.best_price = None  # Best price reached in trade direction
        self.current_size = None  # Track current position size
        
        # Historical tracking
        self.trades = []
        self.trade_signals = []
        
    def has_open_position(self) -> bool:
        return self.open_position is not None

    def set_position(self, direction: str, entry_price: float, size: float):
        self.open_position = direction
        self.best_price = entry_price
        self.current_size = size

    def close_position(self, outcome: str = None, exit_time = None):
        if outcome == 'SL':
            self.last_sl_hit = self.open_position
            self.last_exit_time = exit_time
        self.open_position = None
        self.current_stop_loss = None
        self.current_take_profit = None
        self.best_price = None
        self.current_size = None
    
    def calculate_stop_loss_take_profit(self, direction: str, entry_price: float) -> Tuple[float, float]:
        # ATR-based stop loss
        atr_value = self.atr[0]
        
        # Calculate stop loss distance using ATR
        sl_distance = atr_value * self.p.atr_multiplier
        
        # Calculate stop loss level
        stop_loss = entry_price - sl_distance if direction == 'BUY' else entry_price + sl_distance
        
        # Calculate take profit based on risk:reward ratio
        tp_distance = sl_distance * self.p.risk_reward_ratio
        take_profit = entry_price + tp_distance if direction == 'BUY' else entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def calculate_position_size(self, direction: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on account equity, risk percentage, and stop loss distance
        Returns position size in currency units
        """
        # Get account equity
        equity = self.broker.getvalue()
        
        # Calculate risk amount in account currency
        risk_amount = equity * (self.p.max_risk_percent / 100)
        
        # Calculate stop loss distance in price terms
        sl_distance = abs(entry_price - stop_loss)
        
        # Calculate pip value (assuming 4 decimal places for most forex pairs)
        # For JPY pairs, would need to adjust for 2 decimal places
        pip_value = 0.0001
        
        # Calculate pips at risk
        pips_at_risk = sl_distance / pip_value
        
        # Calculate position size in standard units
        # For a typical forex pair, 1 pip on 1 standard lot (100,000 units) = $10
        # So risk_amount / (pips_at_risk * pip_dollar_value_per_lot) * lot_size = position size
        if pips_at_risk > 0:
            # Value of 1 pip per lot size
            pip_dollar_value = (pip_value * self.p.lot_size) * 10 # $1 per pip for 100,000 unit lot
            
            # Calculate position size
            position_size = (risk_amount / (pips_at_risk * pip_dollar_value)) * self.p.lot_size
            
            # Round down to nearest lot size increment
            position_size = int(position_size / self.p.lot_size) * self.p.lot_size
            
            # Ensure minimum position size
            if position_size < self.p.lot_size:
                position_size = self.p.lot_size
                
            return position_size
        
        return self.p.lot_size  # Default to 1 lot if we can't calculate
    
    def update_trailing_stop(self):
        """Update trailing stop loss based on price movement and ATR"""
        if not self.has_open_position() or self.current_stop_loss is None:
            return False
        
        current_price = self.data.close[0]
        atr_value = self.atr[0]
        move_threshold = atr_value * self.p.trailing_percent
        
        if self.open_position == 'BUY':
            # For buy positions, we track the highest price reached
            if current_price > self.best_price:
                # Price has moved in our favor
                old_best = self.best_price
                self.best_price = current_price
                
                # Only move stop if price moved enough (by trailing_percent of ATR)
                if self.best_price - old_best >= move_threshold:
                    # New stop loss based on new best price and ATR
                    new_stop = self.best_price - (atr_value * self.p.atr_multiplier)
                    
                    # Only move stop if it improves our position
                    if new_stop > self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        return True
                        
        elif self.open_position == 'SELL':
            # For sell positions, we track the lowest price reached
            if current_price < self.best_price:
                # Price has moved in our favor
                old_best = self.best_price
                self.best_price = current_price
                
                # Only move stop if price moved enough (by trailing_percent of ATR)
                if old_best - self.best_price >= move_threshold:
                    # New stop loss based on new best price and ATR
                    new_stop = self.best_price + (atr_value * self.p.atr_multiplier)
                    
                    # Only move stop if it improves our position
                    if new_stop < self.current_stop_loss:
                        self.current_stop_loss = new_stop
                        return True
        
        return False
    
    def analyze_momentum(self) -> float:
        # RSI Analysis
        rsi_signal = SignalType.NEUTRAL
        rsi_value = self.rsi[0]
        
        if rsi_value < 28:
            rsi_signal = SignalType.STRONG_BUY
        elif rsi_value <= 33:
            rsi_signal = SignalType.MID_BUY
        elif rsi_value < 38:
            rsi_signal = SignalType.BUY
        elif rsi_value > 62:
            rsi_signal = SignalType.SELL
        elif rsi_value >= 67:
            rsi_signal = SignalType.MID_SELL
        elif rsi_value > 72:
            rsi_signal = SignalType.STRONG_SELL
            
        # Stochastic Analysis
        stoch_signal = SignalType.NEUTRAL
        stoch_k = self.stoch.percK[0]
        stoch_d = self.stoch.percD[0]
        
        if stoch_k <= 20 and stoch_d <= 20:
            stoch_signal = SignalType.STRONG_BUY
        elif stoch_k >= 80 and stoch_d >= 80:
            stoch_signal = SignalType.STRONG_SELL
            
        # Combined momentum signal
        final_signal = 0.65 * rsi_signal.value + 0.35 * stoch_signal.value
        
        return final_signal
    
    def analyze_trend(self) -> float:
        # EMA Analysis
        ema_signal = SignalType.NEUTRAL
        ema_10_latest = self.ema_short[0]
        ema_20_latest = self.ema_long[0]
        ema_10_prev = self.ema_short[-1]
        ema_20_prev = self.ema_long[-1]
        
        if ema_10_latest > ema_20_latest and ema_10_prev <= ema_20_prev:
            ema_signal = SignalType.STRONG_BUY
        elif ema_10_latest < ema_20_latest and ema_10_prev >= ema_20_prev:
            ema_signal = SignalType.STRONG_SELL
            
        # MACD Analysis
        macd_signal = SignalType.NEUTRAL
        macd = self.macd.macd[0]
        macd_signal_line = self.macd.signal[0]
        
        if macd > macd_signal_line:
            macd_signal = SignalType.BUY
        elif macd < macd_signal_line:
            macd_signal = SignalType.SELL
            
        final_signal = 0.7 * ema_signal.value + 0.3 * macd_signal.value
        
        return final_signal
    
    def analyze_volatility(self) -> float:
        bb_signal = SignalType.NEUTRAL
        close = self.data.close[0]
        bb_lower = self.bbands.lines.bot[0]
        bb_upper = self.bbands.lines.top[0]
        
        if close < bb_lower:
            bb_signal = SignalType.BUY
        elif close > bb_upper:
            bb_signal = SignalType.SELL
            
        return bb_signal.value
    
    def should_trade(self, pair: str) -> Optional[TradeSignal]:
        # Check if there's an open position
        if self.has_open_position():
            return None

        # Get current time
        current_time = self.datetime.datetime(0)
        
        # Check if we're in the hour immediately following a stop loss hit
        if (self.last_sl_hit is not None and 
            self.last_exit_time is not None and 
            current_time - self.last_exit_time <= timedelta(hours=1)):
            
            # Get all signals
            momentum_signal = self.analyze_momentum()
            trend_signal = self.analyze_trend()
            volatility_signal = self.analyze_volatility()
            
            # Calculate weighted signal
            weighted_signal = (self.weights.momentum * momentum_signal + 
                              self.weights.trend * trend_signal + 
                              self.weights.volatility * volatility_signal)
            
            # Determine direction based on signal
            potential_direction = 'BUY' if weighted_signal > 0 else 'SELL'
            
            # If direction matches the last stop loss hit, prevent the trade
            if potential_direction == self.last_sl_hit:
                return None
        
        # Get all signals
        momentum_signal = self.analyze_momentum()
        trend_signal = self.analyze_trend()
        volatility_signal = self.analyze_volatility()
        
        # Calculate weighted signal
        weighted_signal = (self.weights.momentum * momentum_signal + 
                          self.weights.trend * trend_signal + 
                          self.weights.volatility * volatility_signal)
        signal_strength = abs(weighted_signal)
        
        # Check if signal is strong enough
        if signal_strength < self.p.min_strength:
            return None
            
        direction = 'BUY' if weighted_signal > 0 else 'SELL'
        entry_price = self.data.close[0]
        
        # Calculate stop loss and take profit levels
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(direction, entry_price)
        
        # Create trade signal with all information
        trade_signal = TradeSignal(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strength=signal_strength,
            reasons={
                'momentum': f"Momentum signal: {momentum_signal:.2f}",
                'trend': f"Trend signal: {trend_signal:.2f}",
                'volatility': f"Volatility signal: {volatility_signal:.2f}"
            }
        )
        
        return trade_signal
    
    def next(self):
        # Skip if not enough bars are available for calculations
        if len(self) <= max(self.p.ema_long, self.p.macd_slow + self.p.macd_signal):
            return
        
        # Check if we need to update trailing stop for existing position
        if self.has_open_position() and self.p.trailing_atr:
            # If stop was updated, cancel previous stop order and create a new one
            if self.update_trailing_stop():
                # Cancel all pending orders
                for order in self.broker.get_orders_open():
                    self.broker.cancel(order)
                
                # Create new stop loss order
                if self.open_position == 'BUY':
                    self.sell(exectype=bt.Order.Stop, price=self.current_stop_loss, size=self.current_size)
                else:  # SELL
                    self.buy(exectype=bt.Order.Stop, price=self.current_stop_loss, size=self.current_size)
        
        # Get trade signal for new position
        pair = self.data._name if hasattr(self.data, '_name') else "Unknown"
        trade_signal = self.should_trade(pair)
        
        if trade_signal:
            # Calculate position size based on risk management
            position_size = self.calculate_position_size(
                trade_signal.direction, 
                trade_signal.entry_price,
                trade_signal.stop_loss
            )
            
            self.trade_signals.append({
                'datetime': self.datetime.datetime(0),
                'pair': trade_signal.pair,
                'direction': trade_signal.direction,
                'entry_price': trade_signal.entry_price,
                'stop_loss': trade_signal.stop_loss,
                'take_profit': trade_signal.take_profit,
                'strength': trade_signal.strength,
                'position_size': position_size,
                'reasons': trade_signal.reasons
            })
            
            # Store current stop loss and take profit for trailing updates
            self.current_stop_loss = trade_signal.stop_loss
            self.current_take_profit = trade_signal.take_profit
            
            # Create order
            if trade_signal.direction == 'BUY':
                self.buy(size=position_size)
                self.set_position('BUY', trade_signal.entry_price, position_size)
                
                # Set stop loss and take profit orders
                sl_order = self.sell(exectype=bt.Order.Stop, price=trade_signal.stop_loss, size=position_size)
                tp_order = self.sell(exectype=bt.Order.Limit, price=trade_signal.take_profit, size=position_size)
                
                # Link the orders as OCO
                sl_order.addinfo(oco=tp_order)
                tp_order.addinfo(oco=sl_order)
                
            else:  # SELL
                self.sell(size=position_size)
                self.set_position('SELL', trade_signal.entry_price, position_size)
                
                # Set stop loss and take profit orders
                sl_order = self.buy(exectype=bt.Order.Stop, price=trade_signal.stop_loss, size=position_size)
                tp_order = self.buy(exectype=bt.Order.Limit, price=trade_signal.take_profit, size=position_size)
                
                # Link the orders as OCO
                sl_order.addinfo(oco=tp_order)
                tp_order.addinfo(oco=sl_order)
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.open_position == 'SELL':  # This is a closing buy (either TP or SL)
                
                    # Determine if this is a winning trade
                    is_profitable = order.price < self.trade_signals[-1]['entry_price']
                    
                    # Calculate actual profit in dollars
                    entry_price = self.trade_signals[-1]['entry_price']
                    position_size = self.trade_signals[-1]['position_size']
                    pip_value = 0.0001  # Standard pip value for most forex pairs
                    
                    # Calculate pips gained/lost
                    pips = (entry_price - order.price) / pip_value
                    
                    # Calculate dollar value (approximately $1 per pip per 10,000 units)
                    profit_dollars = pips * (position_size / self.p.lot_size) 
                    
                    # Set outcome based on hit of TP or profitable SL hit
                    if order.price <= self.current_take_profit:
                        outcome = 'TP'
                    else:  # SL hit
                        outcome = 'SL_WIN' if is_profitable else 'SL'
                    
                    self.close_position(outcome, self.datetime.datetime(0))
                    
                    # Record trade details
                    self.trades.append({
                        'entry_time': self.trade_signals[-1]['datetime'],
                        'exit_time': self.datetime.datetime(0),
                        'direction': 'SELL',
                        'entry_price': entry_price,
                        'exit_price': order.price,
                        'position_size': position_size,
                        'outcome': outcome,
                        'pips': pips,
                        
                        'profit': profit_dollars
                    })
            else:  # sell order
                if self.open_position == 'BUY':  # This is a closing sell (either TP or SL)
                    
                    # Determine if this is a winning trade
                    is_profitable = order.price > self.trade_signals[-1]['entry_price']
                    
                    # Calculate actual profit in dollars
                    entry_price = self.trade_signals[-1]['entry_price']
                    position_size = self.trade_signals[-1]['position_size']
                    pip_value = 0.0001  # Standard pip value for most forex pairs
                    
                    # Calculate pips gained/lost
                    pips = (order.price - entry_price) / pip_value
                    
                    # Calculate dollar value (approximately $1 per pip per 10,000 units)
                    profit_dollars = pips * (position_size / self.p.lot_size)
                    
                    # Set outcome based on hit of TP or profitable SL hit
                    if order.price >= self.current_take_profit:
                        outcome = 'TP'
                    else:  # SL hit
                        outcome = 'SL_WIN' if is_profitable else 'SL'
                    
                    self.close_position(outcome, self.datetime.datetime(0))
                    
                    # Record trade details
                    self.trades.append({
                        'entry_time': self.trade_signals[-1]['datetime'],
                        'exit_time': self.datetime.datetime(0),
                        'direction': 'BUY',
                        'entry_price': entry_price,
                        'exit_price': order.price,
                        'position_size': position_size,
                        'outcome': outcome,
                        'pips': pips,
                        'profit': profit_dollars
                    })
    
    def stop(self):
        # Calculate and print statistics when backtest is complete
        # Count both TP and SL_WIN as wins
        wins = len([t for t in self.trades if t['outcome'] in ['TP', 'SL_WIN']])
        total_trades = len(self.trades)
        
        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            print(f"\nTotal Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")

def run_backtest(pairs: List[str], start_date: datetime, end_date: datetime):
    """
    Run backtest on multiple currency pairs.
    
    Args:
        pairs: List of currency pairs to analyze
        start_date: Start date for analysis
        end_date: End date for analysis
    """
    results = {}
    
    for pair in pairs:
        
        # Get data
        data = get_forex_data(pair, start_date, end_date)
        if data is None:
            logger.error(f"Failed to fetch data for {pair}")
            continue
       
        # Create cerebro instance
        cerebro = bt.Cerebro()
        
        try:
            # Add data feed - use column names now that we've ensured they're simple strings
            data_feed = bt.feeds.PandasData(
                dataname=data,
                datetime=None,  # Use the index as dates
                open='Open',    # Use 'Open' column 
                high='High',    # Use 'High' column
                low='Low',      # Use 'Low' column
                close='Close',  # Use 'Close' column
                volume='Volume', # Use 'Volume' column
                openinterest=-1 # No open interest
            )
            
            # Add name for identification
            data_feed._name = pair
            cerebro.adddata(data_feed)
            
            # Add the strategy
            cerebro.addstrategy(ForexStrategy)
            
            # Set initial cash
            cerebro.broker.setcash(10000.0)
            
            # Set commission and leverage
            cerebro.broker.setcommission(commission=0.0001, leverage=30) 

            # Run the backtest
            cerebro.run()
            
            # Print final portfolio value
            print(f"Final Portfolio Value for {pair}: ${cerebro.broker.getvalue():.2f}")
            
            # Store results
            results[pair] = cerebro.broker.getvalue()
        except Exception as e:
            logger.error(f"Error running backtest for {pair}: {str(e)}")
            continue
    
    return results


if __name__ == "__main__":
    # Define multiple currency pairs to analyze
    pairs = [
        "AUDUSD", 
        "USDCAD",
        "NZDUSD",
        "EURUSD",
        "GBPUSD",
        "USDCHF",
    ]
    
    end = datetime(2024, 12, 1, 0, 0, 0)
    
    # For 15m timeframe, we need a reasonable amount of historical data
    # Adjust the number of days based on your data availability
    backtest_days = 30
    start = end - timedelta(days=backtest_days)
    
    print(f"Running backtest from {start} to {end}")
    
    # Run backtest
    results = run_backtest(pairs, start, end)
    
    # Print overall results
    total = 0
    print("=" * 40)
    print("\nOverall results :\n")
    for pair, final_value in results.items():
        profit_pct = ((final_value - 10000) / 10000) * 100
        print(f"{pair}: ${final_value:.2f} ({profit_pct:+.2f}%)") 
        total = total + profit_pct
    
    total = round(total, 2)
    print(f"\nTotal Result over all pairs : {total}%")
        
    
