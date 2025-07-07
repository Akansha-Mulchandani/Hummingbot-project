# Hummingbot Adaptive Pure Market Making Strategy

A sophisticated algorithmic trading strategy that combines multiple technical indicators with dynamic spread adjustment, inventory management, and risk controls for optimal market making performance.

## Features

- **Dynamic Spread Adjustment**: Automatically adjusts bid/ask spreads based on market volatility and trends
- **Technical Analysis Integration**: Uses RSI, EMA, and NATR indicators for market analysis
- **Inventory Management**: Maintains balanced portfolio allocation with configurable targets
- **Risk Controls**: Built-in position sizing and stop-loss mechanisms
- **Real-time Monitoring**: Live performance tracking and market analysis display

## Prerequisites

- Docker installed on your system
- Basic understanding of cryptocurrency trading
- Hummingbot Docker image
- Internet connection for market data

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd hummingbot-adaptive-pmm
```

### Step 2: Start Hummingbot Docker Container

**Note**: These instructions have been tested using Command Prompt (cmd) run as Administrator. You can choose to run in normal mode, but Administrator mode is recommended for proper Docker permissions.

```bash
# Create and start the Hummingbot container
docker run -it --name hummingbot -d hummingbot/hummingbot:latest

# Enter the container
docker exec -it hummingbot bash
```

**Windows Users**: 
- Open Command Prompt as Administrator (Right-click CMD → "Run as administrator")
- Or use PowerShell as Administrator if preferred

### Step 3: Install the Strategy Script

```bash
# Create the strategy file
cat > /home/hummingbot/scripts/adaptive_pmm_strategy.py << 'EOF'
# Paste the entire strategy code here
EOF
```

### Step 4: Install Required Dependencies

```bash
# Activate conda environment
conda activate hummingbot

# Install pandas-ta for technical indicators
pip install pandas-ta
```

### Step 5: Configure Exchange Connection

Before running the strategy, you need to configure your exchange connection:

```bash
# Start Hummingbot
python bin/hummingbot.py

# In Hummingbot console, connect to your exchange
connect binance_paper_trade
# Or for live trading: connect binance

# Follow the prompts to enter your API credentials
```

### Step 6: Run the Strategy

```bash
# In Hummingbot console
start --script adaptive_pmm_strategy.py
```

## Configuration

### Key Parameters

You can modify these parameters in the strategy file before running:

```python
# Core Strategy Parameters
trading_pair = "ETH-USDT"          # Trading pair
exchange = "binance_paper_trade"    # Exchange (use binance for live trading)

# Order Parameters
base_order_amount = 0.01           # Base order size
min_order_amount = 0.001           # Minimum order size
max_order_amount = 0.05            # Maximum order size
order_refresh_time = 10            # Order refresh interval (seconds)

# Spread Parameters
min_spread = 0.0005               # Minimum spread (0.05%)
max_spread = 0.01                 # Maximum spread (1%)
base_spread = 0.002               # Base spread (0.2%)

# Risk Management
max_inventory_ratio = 0.7         # Maximum inventory imbalance
inventory_target_ratio = 0.5      # Target inventory ratio
max_position_size = 0.1           # Maximum position size
```

### Technical Indicator Settings

```python
# Volatility Parameters
volatility_scalar = 150           # Volatility impact multiplier
volatility_lookback = 20          # Periods for volatility calculation

# Trend Parameters
rsi_period = 14                   # RSI calculation period
ema_fast = 9                      # Fast EMA period
ema_slow = 21                     # Slow EMA period
trend_strength_threshold = 0.6    # Minimum trend strength
```

## Strategy Logic

### 1. Market Analysis
- **Volatility**: Uses Normalized Average True Range (NATR) to measure market volatility
- **Trend Detection**: Compares fast and slow EMAs to determine trend direction and strength
- **Momentum**: Uses RSI to identify overbought/oversold conditions

### 2. Dynamic Spread Calculation
- Base spread adjusted by volatility, trend strength, and RSI conditions
- Asymmetric spreads during strong trends to capture directional moves
- Automatic spread widening during high volatility periods

### 3. Inventory Management
- Monitors base/quote asset ratio
- Adjusts order sizes to maintain target inventory balance
- Reduces position size when approaching maximum inventory limits

### 4. Risk Controls
- Position size limits based on available balance
- Automatic order cancellation and replacement
- Stop-loss mechanisms for large adverse moves

## Monitoring

The strategy provides real-time status information including:

- Current balances and active orders
- Market analysis (price, volatility, RSI, trend)
- Inventory management status
- Performance metrics (filled orders, estimated PnL)
- Recent market data

Access the status by typing `status` in the Hummingbot console.

## Risk Disclaimer

**⚠️ Important Warning:**

- This strategy is for educational purposes only
- Always test with paper trading first (`binance_paper_trade`)
- Cryptocurrency trading involves significant risk
- Never invest more than you can afford to lose
- Past performance does not guarantee future results

## Troubleshooting

### Common Issues

1. **"pandas_ta not available" error**
   ```bash
   pip install pandas-ta
   ```

2. **Exchange connection failed**
   - Verify API credentials
   - Check internet connection
   - Ensure exchange is supported

3. **Insufficient candle data**
   - Wait for more market data to accumulate
   - Strategy needs at least 21 candles for proper operation

4. **Permission denied when creating file**
   ```bash
   # Ensure you're in the correct directory
   cd /home/hummingbot/scripts/
   
   # Check file permissions
   ls -la
   ```

### Logs

Check strategy logs for detailed debugging:
```bash
# View logs
tail -f logs/hummingbot_logs.log

# Filter for strategy-specific logs
grep "AdaptivePMM" logs/hummingbot_logs.log
```

## Advanced Configuration

### Custom Trading Pairs

To use different trading pairs, modify:
```python
trading_pair = "BTC-USDT"  # or any supported pair
```

### Live Trading Setup

For live trading, change:
```python
exchange = "binance"  # Remove "_paper_trade" suffix
```

**Note**: Ensure you have sufficient balance and have tested thoroughly with paper trading first.

## Performance Optimization

### Recommended Settings by Market Conditions

**High Volatility Markets:**
- Increase `volatility_scalar` to 200-300
- Widen `max_spread` to 0.02-0.03
- Reduce `order_refresh_time` to 5-8 seconds

**Low Volatility Markets:**
- Decrease `volatility_scalar` to 100-150
- Narrow `max_spread` to 0.005-0.01
- Increase `order_refresh_time` to 15-30 seconds

**Trending Markets:**
- Increase `trend_scalar` to 0.5-0.7
- Lower `trend_strength_threshold` to 0.4-0.5
- Adjust `inventory_target_ratio` based on trend direction

## Support

For issues and questions:
- Check Hummingbot documentation: https://docs.hummingbot.org/
- Join Hummingbot Discord: https://discord.gg/hummingbot
- Review strategy logs for error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly with paper trading
4. Submit a pull request with detailed description

---

**Disclaimer**: This software is provided as-is without warranty. Use at your own risk.
