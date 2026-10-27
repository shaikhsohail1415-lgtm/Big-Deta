"""
Generate a synthetic dataset that mimics the schema of the UCI "Online Retail II" data.

Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

This lets the whole project run end-to-end without downloading anything.
For the real project, replace it with the actual dataset (see README.md).

Usage:
    python generate_sample_data.py --customers 1500 --out sample_online_retail.csv
"""
import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ADJ = ["VINTAGE", "RETRO", "HAND PAINTED", "ANTIQUE", "SPOTTY", "STRIPED", "FLORAL", "GLITTER", "WOODEN", "CERAMIC"]
COLOUR = ["RED", "BLUE", "GREEN", "PINK", "IVORY", "BLACK", "YELLOW", "SILVER", "GOLD", "CREAM"]
ITEM = ["MUG", "LANTERN", "CUSHION", "TEA SET", "CANDLE HOLDER", "PHOTO FRAME", "BUNTING", "BAG", "TIN",
        "JAR", "NOTEBOOK", "WRAP", "COASTER", "APRON", "HOOK", "CLOCK", "TRAY", "PLATE", "BOWL", "BOX"]
COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium",
             "Switzerland", "Portugal", "Australia", "Norway", "Italy"]
COUNTRY_P = np.array([0.84, 0.03, 0.03, 0.02, 0.012, 0.012, 0.008, 0.008, 0.006, 0.006, 0.004, 0.004])
COUNTRY_P = COUNTRY_P / COUNTRY_P.sum()

# name: (share of customers, orders range, days-since-last-order range, order-size multiplier)
SEGMENTS = {
    "champions": (0.08, (15, 40), (0, 30), 1.6),
    "loyal":     (0.17, (6, 15),  (10, 60), 1.2),
    "regular":   (0.30, (2, 6),   (30, 150), 1.0),
    "at_risk":   (0.20, (2, 7),   (120, 250), 1.0),
    "lost":      (0.25, (1, 3),   (250, 365), 0.8),
}

START = datetime(2010, 12, 1)
END = datetime(2011, 12, 9)


def build_catalog(rng, n_products=300):
    seen, rows = set(), []
    while len(rows) < n_products:
        desc = f"{random.choice(ADJ)} {random.choice(COLOUR)} {random.choice(ITEM)}"
        if desc in seen:
            continue
        seen.add(desc)
        code = f"{rng.integers(10000, 99999)}"
        price = float(np.round(np.exp(rng.normal(1.0, 0.7)), 2))  # median ~ 2.7
        rows.append((code, desc, max(price, 0.29)))
    return rows


def main(n_customers, out_path, seed):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    catalog = build_catalog(rng)
    # product popularity follows a long-tail distribution
    pop = rng.pareto(1.5, len(catalog)) + 1
    pop = pop / pop.sum()
    bundles = [rng.choice(len(catalog), 3, replace=False) for _ in range(25)]
    hour_p = np.array([1, 3, 5, 6, 7, 8, 7, 6, 5, 3, 1]) / 52

    names = list(SEGMENTS)
    shares = [SEGMENTS[n][0] for n in names]
    rows, invoice_no, total_days = [], 536000, (END - START).days

    for cid in range(12347, 12347 + n_customers):
        seg = rng.choice(names, p=shares)
        _, (o_lo, o_hi), (r_lo, r_hi), mult = SEGMENTS[seg]
        n_orders = int(rng.integers(o_lo, o_hi + 1))
        last_day = max(total_days - int(rng.integers(r_lo, r_hi + 1)), 1)
        first_day = int(rng.integers(0, last_day + 1)) if n_orders > 1 else last_day
        days = sorted(set(rng.integers(first_day, last_day + 1, n_orders - 1).tolist() + [last_day]))
        country = rng.choice(COUNTRIES, p=COUNTRY_P)

        for d in days:
            invoice_no += 1
            ts = START + timedelta(days=int(d), hours=int(rng.choice(range(8, 19), p=hour_p)),
                                   minutes=int(rng.integers(0, 60)))
            n_lines = max(1, int(rng.poisson(6 * mult)))
            picks = set(rng.choice(len(catalog), n_lines, p=pop).tolist())
            if rng.random() < 0.3:  # bundle -> creates real association rules
                picks.update(bundles[int(rng.integers(0, len(bundles)))].tolist())
            for p in picks:
                code, desc, price = catalog[p]
                qty = int(max(1, rng.poisson(5 * mult)))
                rows.append([str(invoice_no), code, desc, qty, ts, price, cid, country])
            if rng.random() < 0.02:  # a cancellation
                code, desc, price = catalog[int(rng.integers(0, len(catalog)))]
                rows.append([f"C{invoice_no}", code, desc, -int(rng.integers(1, 5)), ts, price, cid, country])

    # guest checkouts (no CustomerID) - realistic dirty data
    for _ in range(int(n_customers * 1.5)):
        invoice_no += 1
        ts = START + timedelta(days=int(rng.integers(0, total_days)), hours=int(rng.integers(8, 19)))
        for p in rng.choice(len(catalog), int(rng.integers(1, 6)), p=pop):
            code, desc, price = catalog[int(p)]
            rows.append([str(invoice_no), code, desc, int(rng.integers(1, 8)), ts, price, None, "United Kingdom"])

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    # a few zero-price rows
    bad = df.sample(frac=0.002, random_state=seed).index
    df.loc[bad, "UnitPrice"] = 0.0
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    df["InvoiceDate"] = df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows, {df['CustomerID'].nunique():,} customers -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=1500)
    ap.add_argument("--out", default="sample_online_retail.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.customers, a.out, a.seed)
