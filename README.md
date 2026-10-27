# E-commerce Sales & Customer Segmentation with Apache Spark

Big data project: PySpark cleans e-commerce transactions, Spark SQL analyses sales, Spark MLlib
K-Means segments customers using RFM, FP-Growth finds frequently bought-together products, and a
Streamlit dashboard presents the results.

## Files (everything is in one flat folder)

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (reads the result CSVs) |
| `pipeline.py` | PySpark pipeline: ingest, clean, Spark SQL, RFM, K-Means, FP-Growth |
| `generate_sample_data.py` | Creates the synthetic sample dataset |
| `sample_online_retail.csv` | Synthetic sample data (about 79k rows, 1,500 customers) |
| `kpis.csv`, `monthly_revenue.csv`, `top_products.csv`, `country_revenue.csv`, `hourly_sales.csv`, `rfm_segments.csv`, `cluster_profile.csv`, `k_selection.csv`, `association_rules.csv` | Precomputed pipeline results used by the dashboard |
| `requirements.txt` | Dashboard dependencies (used by Streamlit Cloud) |
| `requirements-spark.txt` | Extra dependencies to run the pipeline |

## Pipeline

```
raw CSV -> PySpark ingest -> clean -> Spark SQL aggregates --+
                               |                             +--> result CSVs --> Streamlit app
                               +--> RFM -> K-Means ----------+
                               +--> FP-Growth rules ---------+
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Dashboard only (uses the included result CSVs)
pip install -r requirements.txt
streamlit run app.py

# Re-run the Spark pipeline (needs Java 17 or 21)
pip install -r requirements-spark.txt
python pipeline.py --input sample_online_retail.csv --k 4
```

Options: `--k` (clusters), `--k_max` (max k for the silhouette search), `--skip_basket`, `--save_parquet`,
`--output` (folder for the result CSVs, default is the current folder).

## Use a real dataset

Export it as CSV with these columns: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`
(`InvoiceDate` like `2010-12-01 08:26:00`), for example the UCI Online Retail II dataset, then run:

```bash
python pipeline.py --input online_retail_II.csv --k 4
```

GitHub rejects files over 100 MB, so do not commit very large raw datasets.

## Google Colab

```python
!pip install pyspark pandas==2.2.2
# upload the files, then:
!python pipeline.py --input sample_online_retail.csv
```

## Deploy on Streamlit Community Cloud

1. Upload all these files to the root of a GitHub repo (the result CSVs must be included).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create app, choose the repo and branch, set the main file to `app.py`, and deploy.

Only `requirements.txt` is needed in the cloud, so Java and Spark are not required there.

## Results on the sample data (synthetic, for demonstration)

| Segment | Customers | Avg recency (days) | Avg orders | Avg spend | Revenue share |
|---|---|---|---|---|---|
| Champions | 115 | 17 | 25.7 | 6,139 | 48.7% |
| Loyal Customers | 316 | 36 | 9.5 | 1,323 | 28.8% |
| At-Risk | 656 | 125 | 4.1 | 420 | 19.0% |
| Lost / Hibernating | 413 | 284 | 1.8 | 124 | 3.5% |

Re-run the pipeline on a real dataset and replace this table with your own results.
