import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import matplotlib.pyplot as plt

# --- Config ---
TICKER = "IS0P.DE"
START  = "2010-01-01"
CASH_PER_MONTH = 100.0
FEES_BP = 0.0 

# --- Data ---
data = yf.download(TICKER, start=START, progress=False)
price = data["Close"].squeeze()
print(data)
# --- Monthly schedule (last trading day of each month) ---
month_ends = price.resample("ME").last().index
entries = price.index.isin(month_ends)  # this returns a numpy array

# Convert entries to a pandas Series aligned with price.index
entries = pd.Series(entries, index=price.index)

# --- Convert $100 to shares on each entry date (fractional shares allowed) ---
shares_to_buy = pd.Series(0.0, index=price.index)
shares_to_buy.loc[entries] = CASH_PER_MONTH / price[entries]

# --- Build the portfolio ---

pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=None,                # never sell; pure accumulation
    size=shares_to_buy,        # buy this many shares on each signal
    fees=FEES_BP / 10000,      # basis points -> fraction (0.001 = 0.1%)
    slippage=0.0,
    init_cash=1_000_000.0,     # large to avoid cash shortfall (we’ll track contributions separately)
    freq="D"
)

# --- Stats / outputs ---

contrib = pd.Series(0.0, index=price.index)
contrib[entries] = CASH_PER_MONTH
total_contributions = contrib.sum()

summary = pf.stats()
print("=== VectorBT Stats ===")
print(summary)

print("\n=== Contributions ===")
print(f"Total contributed: ${total_contributions:,.2f}")
print(f"Number of buys: {int(entries.sum())}")

final_value = pf.value().iloc[-1]
pnl_vs_contrib = final_value - (1_000_000.0)  # value minus initial cash
# Adjust to show gain vs contributions:
# Since we faked a big init cash, subtract the unused cash to get “asset value”
shares_held = pf.positions.records_readable["Size"].sum()  # or pf.asset_flow().cumsum().iloc[-1] / price.iloc[-1]
asset_value = shares_held * price.iloc[-1]
gain_vs_contrib = asset_value - total_contributions

print("\n=== Ending Snapshot (Approximate) ===")
print(f"ETF price:          ${price.iloc[-1]:.2f}")
print(f"Shares held:        {shares_held:,.6f}")
print(f"Asset value:        ${asset_value:,.2f}")
print(f"Contributions:      ${total_contributions:,.2f}")
print(f"Gain vs contribution: ${gain_vs_contrib:,.2f}")

# Optional: money-weighted return (XIRR) from dated cash flows
# Positive = withdrawals (value when you sell), negative = deposits (your buys)
# Here we simulate deposits of -$100 on buy dates and one terminal withdrawal = +asset_value
try:
    from numpy_financial import xirr  # pip install numpy-financial
    cashflows = []
    dates = []
    for dt, val in contrib[contrib != 0].items():
        cashflows.append(-val)
        dates.append(dt)
    cashflows.append(asset_value)
    dates.append(price.index[-1])
    irr = xirr(pd.Series(cashflows, index=pd.to_datetime(dates)))
    print(f"\nApprox. money-weighted return (XIRR): {irr:.2%}")
except Exception:
    print("\n(Install numpy-financial if you want XIRR: `pip install numpy-financial`)")

# --- Plot (quick look) ---
pf.value().plot(title=f"{TICKER} DCA Portfolio Value"); plt.show()