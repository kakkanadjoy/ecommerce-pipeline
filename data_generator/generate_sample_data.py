"""
Generates simple synthetic e-commerce data to practice a bronze/silver/gold
Databricks pipeline. Run this locally to produce CSV files that simulate a
source system dropping daily order extracts.

Usage:
    python generate_sample_data.py --day 2026-08-25 --num_orders 200

Each run appends ONE new day's worth of order data as a new CSV file,
simulating incremental daily ingestion. Customers and products are generated
once and reused so silver-layer joins have something to key against.
"""

import argparse
import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

PRODUCTS = [
    (1, "Wireless Mouse", "Electronics", 24.99),
    (2, "USB-C Cable", "Electronics", 9.99),
    (3, "Notebook", "Office", 4.50),
    (4, "Desk Lamp", "Office", 19.99),
    (5, "Water Bottle", "Lifestyle", 14.99),
    (6, "Backpack", "Lifestyle", 39.99),
    (7, "Coffee Mug", "Kitchen", 8.99),
    (8, "Headphones", "Electronics", 59.99),
]

FIRST_NAMES = ["Alex", "Jamie", "Sam", "Priya", "Wei", "Diego", "Amara", "Liam"]
LAST_NAMES = ["Smith", "Chen", "Patel", "Garcia", "Kim", "Johnson", "Nguyen"]


def ensure_customers(path, num_customers=50):
    """Write customers.csv once — acts like a slow-changing reference table."""
    filepath = os.path.join(path, "customers.csv")
    if os.path.exists(filepath):
        return
    os.makedirs(path, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["customer_id", "first_name", "last_name", "signup_date", "region"])
        regions = ["US-East", "US-West", "EU", "APAC"]
        for cid in range(1, num_customers + 1):
            fn = random.choice(FIRST_NAMES)
            ln = random.choice(LAST_NAMES)
            signup = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 500))
            writer.writerow([cid, fn, ln, signup.strftime("%Y-%m-%d"), random.choice(regions)])
    print(f"Wrote {filepath}")


def ensure_products(path):
    """Write products.csv once — small reference table."""
    filepath = os.path.join(path, "products.csv")
    if os.path.exists(filepath):
        return
    os.makedirs(path, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "product_name", "category", "unit_price"])
        for row in PRODUCTS:
            writer.writerow(row)
    print(f"Wrote {filepath}")


def generate_orders_for_day(path, day, num_orders, num_customers=50):
    """
    Writes ONE file: orders_<day>.csv — simulates a new daily batch landing
    in a raw folder, the way a real source system export would arrive.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"orders_{day}.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "customer_id", "product_id", "quantity", "order_date", "order_ts"])
        base_order_id = int(datetime.strptime(day, "%Y-%m-%d").strftime("%Y%m%d")) * 1000
        for i in range(num_orders):
            order_id = base_order_id + i
            customer_id = random.randint(1, num_customers)
            product_id = random.choice(PRODUCTS)[0]
            quantity = random.randint(1, 4)
            order_ts = datetime.strptime(day, "%Y-%m-%d") + timedelta(
                hours=random.randint(0, 23), minutes=random.randint(0, 59)
            )
            writer.writerow([order_id, customer_id, product_id, quantity, day, order_ts.isoformat()])
    print(f"Wrote {filepath} ({num_orders} orders)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=str, default=datetime.today().strftime("%Y-%m-%d"),
                         help="Date for this batch of orders, format YYYY-MM-DD")
    parser.add_argument("--num_orders", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="./raw_data")
    args = parser.parse_args()

    ensure_customers(args.out_dir)
    ensure_products(args.out_dir)
    generate_orders_for_day(args.out_dir, args.day, args.num_orders)
