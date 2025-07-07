
import logging
from decimal import Decimal
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent, BuyOrderCompletedEvent, SellOrderCompletedEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase


class AdaptivePMMStrategy(ScriptStrategyBase):
    """
    Adaptive Pure Market Making Strategy
    
    A sophisticated algorithmic trading strategy that combines multiple technical indicators
    with dynamic spread adjustment, inventory management, and risk controls for optimal
    market making performance.
    """
    
    # Core Strategy Parameters
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    
    # Order Parameters
    base_order_amount = 0.01
    min_order_amount = 0.001
    max_order_amount = 0.05
    order_refresh_time = 10
    
    # Spread Parameters
    min_spread = 0.0005
    max_spread = 0.01
    base_spread = 0.002
    
    # Volatility Parameters
    volatility_scalar = 150
    volatility_lookback = 20
    
    # Trend Parameters
    trend_strength_threshold = 0.6
    trend_scalar = 0.3
    rsi_period = 14
    ema_fast = 9
    ema_slow = 21
    
    # Inventory Management
    max_inventory_ratio = 0.7
    inventory_target_ratio = 0.5
    inventory_scalar = 2.0
    
    # Risk Management
    max_position_size = 0.1
    stop_loss_threshold = 0.02
    
    # Candle Configuration
    candle_exchange = "binance"
    candles_interval = "1m"
    max_records = 500
    
    # Internal State Variables
    create_timestamp = 0
    last_inventory_check = 0
    inventory_check_interval = 30
    total_pnl = Decimal("0")
    filled_orders_count = 0
    
    # Initialize candles feed
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))
    
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        self.logger().info("AdaptivePMMStrategy initialized successfully")
    
    def on_stop(self):
        self.candles.stop()
        self.logger().info("AdaptivePMMStrategy stopped")
    
    def on_tick(self):
        try:
            if not self.ready_to_trade:
                return
                
            if not self.connectors.get(self.exchange) or not self.connectors[self.exchange].ready:
                self.logger().warning("Exchange connector not ready")
                return
                
            current_time = self.current_timestamp
            
            if current_time >= self.create_timestamp:
                self.execute_strategy()
                
            if current_time >= self.last_inventory_check + self.inventory_check_interval:
                self.manage_inventory()
                self.last_inventory_check = current_time
                
        except Exception as e:
            self.logger().error(f"Error in on_tick: {str(e)}")
            self.create_timestamp = self.current_timestamp + 60
    
    def execute_strategy(self):
        try:
            self.cancel_all_orders()
            
            market_data = self.get_market_analysis()
            if not market_data:
                self.logger().warning("Insufficient market data, skipping this cycle")
                return
            
            proposal = self.create_proposal(market_data)
            if proposal:
                proposal_adjusted = self.adjust_proposal_to_budget(proposal)
                self.place_orders(proposal_adjusted)
            
            volatility = market_data.get('volatility', 0)
            adaptive_refresh = max(5, self.order_refresh_time * (1 - volatility))
            self.create_timestamp = self.current_timestamp + adaptive_refresh
            
        except Exception as e:
            self.logger().error(f"Error in execute_strategy: {str(e)}")
    
    def get_market_analysis(self) -> Optional[Dict]:
        try:
            if not self.candles.ready:
                self.logger().info("Candles feed not ready yet, skipping analysis")
                return None
                
            if self.candles.candles_df is None or len(self.candles.candles_df) < max(self.rsi_period, self.ema_slow, self.volatility_lookback):
                self.logger().info(f"Insufficient candle data: {len(self.candles.candles_df) if self.candles.candles_df is not None else 0} candles")
                return None
            
            df = self.candles.candles_df.copy()
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger().error(f"Missing required columns in candle data: {df.columns.tolist()}")
                return None
            
            df = self.calculate_technical_indicators(df)
            
            if len(df) == 0:
                self.logger().warning("No data after calculating indicators")
                return None
            
            latest = df.iloc[-1]
            
            volatility_col = f'NATR_{self.volatility_lookback}'
            volatility = float(latest[volatility_col]) if volatility_col in df.columns and not pd.isna(latest[volatility_col]) else 0.01
            
            rsi_col = f'RSI_{self.rsi_period}'
            rsi = float(latest[rsi_col]) if rsi_col in df.columns and not pd.isna(latest[rsi_col]) else 50
            
            ema_fast_col = f'EMA_{self.ema_fast}'
            ema_slow_col = f'EMA_{self.ema_slow}'
            
            ema_fast = float(latest[ema_fast_col]) if ema_fast_col in df.columns and not pd.isna(latest[ema_fast_col]) else float(latest['close'])
            ema_slow = float(latest[ema_slow_col]) if ema_slow_col in df.columns and not pd.isna(latest[ema_slow_col]) else float(latest['close'])
            
            trend_direction = 1 if ema_fast > ema_slow else -1
            trend_strength = abs(ema_fast - ema_slow) / ema_slow if ema_slow != 0 else 0
            
            rsi_overbought = rsi > 70
            rsi_oversold = rsi < 30
            rsi_neutral = 30 <= rsi <= 70
            
            return {
                'volatility': volatility,
                'rsi': rsi,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'rsi_neutral': rsi_neutral,
                'current_price': float(latest['close'])
            }
            
        except Exception as e:
            self.logger().error(f"Error in market analysis: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not hasattr(df, 'ta'):
                self.logger().error("pandas_ta not available")
                return df
            
            min_periods = max(self.rsi_period, self.ema_slow, self.volatility_lookback)
            if len(df) < min_periods:
                self.logger().warning(f"Insufficient data for indicators: {len(df)} < {min_periods}")
                return df
            
            try:
                df.ta.natr(length=self.volatility_lookback, append=True)
            except Exception as e:
                self.logger().warning(f"Error calculating NATR: {e}")
                df[f'NATR_{self.volatility_lookback}'] = df['high'].rolling(self.volatility_lookback).std() / df['close'] * 100
            
            try:
                df.ta.rsi(length=self.rsi_period, append=True)
            except Exception as e:
                self.logger().warning(f"Error calculating RSI: {e}")
                df[f'RSI_{self.rsi_period}'] = 50
            
            try:
                df.ta.ema(length=self.ema_fast, append=True)
                df.ta.ema(length=self.ema_slow, append=True)
            except Exception as e:
                self.logger().warning(f"Error calculating EMAs: {e}")
                df[f'EMA_{self.ema_fast}'] = df['close'].rolling(self.ema_fast).mean()
                df[f'EMA_{self.ema_slow}'] = df['close'].rolling(self.ema_slow).mean()
            
            return df
            
        except Exception as e:
            self.logger().error(f"Error calculating indicators: {str(e)}")
            return df
    
    def create_proposal(self, market_data: Dict) -> List[OrderCandidate]:
        try:
            current_price = self.get_current_price()
            if not current_price:
                return []
            
            spreads = self.calculate_dynamic_spreads(market_data)
            order_sizes = self.calculate_order_sizes(market_data)
            
            bid_price = current_price * (Decimal("1") - Decimal(str(spreads['bid_spread'])))
            ask_price = current_price * (Decimal("1") + Decimal(str(spreads['ask_spread'])))
            
            best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
            best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
            
            bid_price = min(bid_price, best_bid * Decimal("0.9999"))
            ask_price = max(ask_price, best_ask * Decimal("1.0001"))
            
            orders = []
            
            if order_sizes['buy_size'] > 0:
                buy_order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.BUY,
                    amount=Decimal(str(order_sizes['buy_size'])),
                    price=bid_price
                )
                orders.append(buy_order)
            
            if order_sizes['sell_size'] > 0:
                sell_order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.SELL,
                    amount=Decimal(str(order_sizes['sell_size'])),
                    price=ask_price
                )
                orders.append(sell_order)
            
            return orders
            
        except Exception as e:
            self.logger().error(f"Error creating proposal: {str(e)}")
            return []
    
    def calculate_dynamic_spreads(self, market_data: Dict) -> Dict[str, float]:
        try:
            base_spread = self.base_spread
            
            volatility = market_data.get('volatility', 0)
            volatility_adjustment = volatility * self.volatility_scalar
            
            trend_direction = market_data.get('trend_direction', 0)
            trend_strength = market_data.get('trend_strength', 0)
            
            rsi = market_data.get('rsi', 50)
            rsi_adjustment = 0
            
            if market_data.get('rsi_overbought', False):
                rsi_adjustment = 0.2
            elif market_data.get('rsi_oversold', False):
                rsi_adjustment = 0.2
            
            if trend_direction > 0 and trend_strength > self.trend_strength_threshold:
                bid_spread = base_spread + volatility_adjustment + rsi_adjustment + (trend_strength * self.trend_scalar)
                ask_spread = base_spread + volatility_adjustment + rsi_adjustment - (trend_strength * self.trend_scalar * 0.5)
            elif trend_direction < 0 and trend_strength > self.trend_strength_threshold:
                bid_spread = base_spread + volatility_adjustment + rsi_adjustment - (trend_strength * self.trend_scalar * 0.5)
                ask_spread = base_spread + volatility_adjustment + rsi_adjustment + (trend_strength * self.trend_scalar)
            else:
                bid_spread = ask_spread = base_spread + volatility_adjustment + rsi_adjustment
            
            bid_spread = max(self.min_spread, min(self.max_spread, bid_spread))
            ask_spread = max(self.min_spread, min(self.max_spread, ask_spread))
            
            return {
                'bid_spread': float(bid_spread),
                'ask_spread': float(ask_spread)
            }
            
        except Exception as e:
            self.logger().error(f"Error calculating spreads: {str(e)}")
            return {'bid_spread': float(self.base_spread), 'ask_spread': float(self.base_spread)}
    
    def calculate_order_sizes(self, market_data: Dict) -> Dict[str, float]:
        try:
            inventory_ratio = self.get_inventory_ratio()
            base_size = self.base_order_amount
            
            volatility = market_data.get('volatility', 0)
            volatility_scale = max(0.5, 1 - volatility * 2)
            
            inventory_imbalance = inventory_ratio - self.inventory_target_ratio
            
            if inventory_imbalance > 0.1:
                buy_size = base_size * volatility_scale * (1 - inventory_imbalance * self.inventory_scalar)
                sell_size = base_size * volatility_scale * (1 + inventory_imbalance * self.inventory_scalar)
            elif inventory_imbalance < -0.1:
                buy_size = base_size * volatility_scale * (1 + abs(inventory_imbalance) * self.inventory_scalar)
                sell_size = base_size * volatility_scale * (1 - abs(inventory_imbalance) * self.inventory_scalar)
            else:
                buy_size = sell_size = base_size * volatility_scale
            
            buy_size = max(self.min_order_amount, min(self.max_order_amount, buy_size))
            sell_size = max(self.min_order_amount, min(self.max_order_amount, sell_size))
            
            return {
                'buy_size': float(buy_size),
                'sell_size': float(sell_size)
            }
            
        except Exception as e:
            self.logger().error(f"Error calculating order sizes: {str(e)}")
            return {'buy_size': float(self.base_order_amount), 'sell_size': float(self.base_order_amount)}
    
    def get_inventory_ratio(self) -> float:
        try:
            base_balance = self.connectors[self.exchange].get_available_balance(self.trading_pair.split("-")[0])
            quote_balance = self.connectors[self.exchange].get_available_balance(self.trading_pair.split("-")[1])
            current_price = self.get_current_price()
            
            if not current_price or not base_balance or not quote_balance:
                return 0.5
            
            total_value = base_balance * current_price + quote_balance
            if total_value == 0:
                return 0.5
            
            base_value_ratio = (base_balance * current_price) / total_value
            return float(base_value_ratio)
            
        except Exception as e:
            self.logger().error(f"Error calculating inventory ratio: {str(e)}")
            return 0.5
    
    def get_current_price(self) -> Optional[Decimal]:
        try:
            return self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        except Exception as e:
            self.logger().error(f"Error getting current price: {str(e)}")
            return None
    
    def manage_inventory(self):
        try:
            inventory_ratio = self.get_inventory_ratio()
            
            if inventory_ratio > self.max_inventory_ratio:
                self.logger().warning(f"Inventory too high: {inventory_ratio:.2%}")
                
            elif inventory_ratio < (1 - self.max_inventory_ratio):
                self.logger().warning(f"Inventory too low: {inventory_ratio:.2%}")
                
        except Exception as e:
            self.logger().error(f"Error in inventory management: {str(e)}")
    
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        try:
            return self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)
        except Exception as e:
            self.logger().error(f"Error adjusting proposal to budget: {str(e)}")
            return proposal
    
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        try:
            for order in proposal:
                self.place_order(self.exchange, order)
        except Exception as e:
            self.logger().error(f"Error placing orders: {str(e)}")
    
    def place_order(self, connector_name: str, order: OrderCandidate):
        try:
            if order.order_side == TradeType.SELL:
                self.sell(connector_name=connector_name, trading_pair=order.trading_pair, 
                         amount=order.amount, order_type=order.order_type, price=order.price)
            elif order.order_side == TradeType.BUY:
                self.buy(connector_name=connector_name, trading_pair=order.trading_pair, 
                        amount=order.amount, order_type=order.order_type, price=order.price)
        except Exception as e:
            self.logger().error(f"Error placing individual order: {str(e)}")
    
    def cancel_all_orders(self):
        try:
            for order in self.get_active_orders(connector_name=self.exchange):
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
        except Exception as e:
            self.logger().error(f"Error cancelling orders: {str(e)}")
    
    def did_fill_order(self, event: OrderFilledEvent):
        try:
            self.filled_orders_count += 1
            
            msg = (f"{event.trade_type.name} {event.amount:.4f} {event.trading_pair} "
                   f"at {event.price:.4f} | Fill #{self.filled_orders_count}")
            
            self.log_with_clock(logging.INFO, msg)
            self.notify_hb_app_with_timestamp(msg)
            
            if event.trade_type == TradeType.SELL:
                self.total_pnl += event.amount * event.price
            else:
                self.total_pnl -= event.amount * event.price
                
        except Exception as e:
            self.logger().error(f"Error handling order fill: {str(e)}")
    
    def format_status(self) -> str:
        try:
            if not self.ready_to_trade:
                return "Market connectors are not ready."
            
            lines = []
            
            balance_df = self.get_balance_df()
            lines.extend(["", "Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])
            
            try:
                df = self.active_orders_df()
                lines.extend(["", "Active Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
            except ValueError:
                lines.extend(["", "No active orders."])
            
            market_data = self.get_market_analysis()
            if market_data:
                lines.extend(["\n" + "="*60])
                lines.extend(["Market Analysis:"])
                lines.extend([f"    Current Price: ${market_data['current_price']:.4f}"])
                lines.extend([f"    Volatility (NATR): {market_data['volatility']:.4f}"])
                lines.extend([f"    RSI: {market_data['rsi']:.2f}"])
                lines.extend([f"    Trend: {'Upward' if market_data['trend_direction'] > 0 else 'Downward'} "
                            f"(Strength: {market_data['trend_strength']:.3f})"])
            
            inventory_ratio = self.get_inventory_ratio()
            lines.extend(["\n" + "="*60])
            lines.extend(["Inventory Management:"])
            lines.extend([f"    Current Ratio: {inventory_ratio:.2%}"])
            lines.extend([f"    Target Ratio: {self.inventory_target_ratio:.2%}"])
            lines.extend([f"    Status: {'Balanced' if abs(inventory_ratio - self.inventory_target_ratio) < 0.1 else 'Imbalanced'}"])
            
            lines.extend(["\n" + "="*60])
            lines.extend(["Performance:"])
            lines.extend([f"    Orders Filled: {self.filled_orders_count}"])
            lines.extend([f"    Estimated PnL: ${self.total_pnl:.2f}"])
            
            if len(self.candles.candles_df) > 0:
                lines.extend(["\n" + "="*60])
                lines.extend([f"Recent Candles ({self.candles.name} - {self.candles.interval}):"])
                recent_candles = self.candles.candles_df.tail(3)
                lines.extend(["    " + line for line in recent_candles.to_string(index=False).split("\n")])
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger().error(f"Error formatting status: {str(e)}")
            return "Error displaying status - check logs for details."
