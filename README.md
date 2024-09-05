# Consolidation Finder Bot

This repository contains a Python-based bot that identifies consolidation patterns in cryptocurrency market data using the Binance WebSocket API. The bot automatically detects consolidations, opens trades based on the detected patterns, and generates candlestick charts highlighting the consolidations.

## Features

- **Real-time Data Streaming**: Uses Binance WebSocket API to receive real-time market data.
- **Consolidation Detection**: Analyzes candlestick data to identify consolidation patterns based on peak analysis.
- **Automated Trading**: Opens trades based on detected consolidations, with predefined entry, stop-loss, and take-profit levels.
- **Data Visualization**: Generates and saves candlestick charts with highlighted consolidation zones.

## Requirements

- Python 3.7+
- Binance API
- NumPy
- Pandas
- Matplotlib
- mplfinance
- websockets
- asyncio

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/8bocian/ConsolidationFinder.git
   cd ConsolidationFinder
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have set up your Python environment with the required packages.
2. Run the bot with the following command:

   ```bash
   python bot.py
   ```

3. The bot will connect to the Binance WebSocket API, start receiving data, and begin analyzing it for consolidation patterns.
4. Trade decisions and generated charts will be saved in the `trades` directory.

## Configuration

- **Trading Pair**: The default trading pair is set to `BTCUSDT`. This can be changed by modifying the `pair` variable in the script.
- **Trade Size**: The default trade size is set to $10,000. This can be adjusted by changing the `size` parameter in the `Trade` class.

## How It Works

1. **Data Collection**: The bot collects market data in real-time from the Binance WebSocket API.
2. **Consolidation Detection**: It identifies consolidation zones by analyzing peaks and troughs within a specified window of candlesticks.
3. **Trade Execution**: When a consolidation pattern is identified, the bot opens a trade with defined entry, stop-loss, and take-profit levels.
4. **Visualization**: A candlestick chart with the detected consolidation is saved as a PNG file in the `trades` directory.

## Warning and Disclaimer

- This bot is intended for educational purposes only and should not be used in live trading without thorough testing and modifications.
- Trading cryptocurrencies involves significant risk and can result in substantial financial loss. Use this bot at your own risk.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or reach out to the repository owner at [oskar.mozdzen@gmail.com].
