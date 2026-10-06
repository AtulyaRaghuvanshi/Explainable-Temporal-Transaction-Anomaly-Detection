"""Generate a small IEEE-CIS-shaped dataset (for demos/tests when the Kaggle data is unavailable).

Same file names and key columns as the real competition files, with a planted fraud signal
(night-time, large amounts, new cards, certain email domains). Writes to data/raw/.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument('--rows', type=int, default=60000); ap.add_argument('--seed', type=int, default=42); a = ap.parse_args()
rng = np.random.default_rng(a.seed); n = a.rows
dt = np.sort(rng.integers(86400, 86400 * 180, n))
card1 = rng.zipf(1.3, n) % 3000 + 1000
amt = np.round(rng.lognormal(3.6, 1.1, n), 2)
hour = (dt // 3600) % 24
domains = np.array(['gmail.com', 'yahoo.com', 'hotmail.com', 'anonymous.com', 'outlook.com', 'aol.com', None], dtype=object)
dom = domains[rng.choice(len(domains), n, p=[.35, .2, .12, .05, .1, .08, .10])]
first_seen = ~pd.Series(card1).duplicated().values
logit = -5.3 + 0.9 * (hour < 6) + 0.5 * np.log1p(amt) / 2 + 1.1 * first_seen + 1.2 * (dom == 'anonymous.com')
fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
tx = pd.DataFrame({'TransactionID': np.arange(2987000, 2987000 + n), 'isFraud': fraud, 'TransactionDT': dt, 'TransactionAmt': amt,
                   'ProductCD': rng.choice(['W', 'C', 'R', 'H', 'S'], n, p=[.7, .1, .08, .07, .05]),
                   'card1': card1, 'card2': rng.integers(100, 600, n).astype(float), 'card3': 150.0,
                   'card4': rng.choice(['visa', 'mastercard', 'discover', 'american express'], n, p=[.6, .3, .05, .05]),
                   'card5': rng.integers(100, 240, n).astype(float), 'card6': rng.choice(['debit', 'credit'], n, p=[.75, .25]),
                   'addr1': rng.integers(100, 500, n).astype(float), 'addr2': 87.0, 'P_emaildomain': dom})
for c in [f'C{i}' for i in range(1, 7)] + [f'D{i}' for i in range(1, 6)] + [f'V{i}' for i in range(1, 21)]:
    tx[c] = np.where(rng.random(n) < .15, np.nan, rng.poisson(2, n) + fraud * rng.poisson(1, n))
ident_ids = tx.TransactionID.values[rng.random(n) < .25]; m = len(ident_ids)
ident = pd.DataFrame({'TransactionID': ident_ids, 'id_01': rng.normal(-5, 8, m).round(1), 'id_02': rng.integers(1000, 500000, m).astype(float),
                      'DeviceType': rng.choice(['desktop', 'mobile'], m), 'DeviceInfo': rng.choice(['Windows', 'iOS Device', 'MacOS', 'SM-G960F', 'Trident/7.0'], m)})
out = root / 'data' / 'raw'; out.mkdir(parents=True, exist_ok=True)
tx.to_csv(out / 'train_transaction.csv', index=False); ident.to_csv(out / 'train_identity.csv', index=False)
print(f'wrote {n} transactions ({fraud.mean()*100:.2f}% fraud) and {m} identity rows to {out}')
