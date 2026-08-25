"""
Unit tests for silver_transform.py.

These tests run against a LOCAL Spark session (no Databricks cluster, no
Unity Catalog, no real tables needed) — this is exactly why the pure
transformation functions were kept separate from I/O. We're testing business
logic (dedup rules, referential integrity, validation) in complete isolation.

Run locally with:  pytest tests/test_silver_transform.py -v
This is also what GitHub Actions CI runs automatically on every pull request.
"""

from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from src.transformations.silver_transform import (
    clean_customers,
    clean_products,
    clean_orders,
)


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("silver_transform_tests")
        .getOrCreate()
    )


def test_clean_customers_deduplicates_and_keeps_latest(spark):
    data = [
        (1, "  Alex ", "Smith", "2025-01-01", "US-East", datetime(2026, 8, 20)),
        (1, "Alex", "Smith-Jones", "2025-01-01", "US-East", datetime(2026, 8, 25)),  # newer, should win
        (2, "Priya", "Patel", "2025-02-01", "APAC", datetime(2026, 8, 20)),
    ]
    columns = ["customer_id", "first_name", "last_name", "signup_date", "region", "_ingested_at"]
    df = spark.createDataFrame(data, columns)

    result = clean_customers(df).orderBy("customer_id").collect()

    assert len(result) == 2  # customer_id=1 deduplicated down to one row
    assert result[0]["last_name"] == "Smith-Jones"  # the newer row won
    assert result[0]["first_name"] == "Alex"  # whitespace was trimmed


def test_clean_customers_drops_null_ids(spark):
    from pyspark.sql.types import (
        StructType, StructField, IntegerType, StringType, TimestampType,
    )

    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("signup_date", StringType(), True),
        StructField("region", StringType(), True),
        StructField("_ingested_at", TimestampType(), True),
    ])
    data = [(None, "Ghost", "Row", "2025-01-01", "US-East", datetime(2026, 8, 20))]
    df = spark.createDataFrame(data, schema)

    result = clean_customers(df).collect()
    assert len(result) == 0


def test_clean_products_drops_non_positive_price(spark):
    data = [
        (1, "Widget", "Misc", 9.99, datetime(2026, 8, 20)),
        (2, "Broken Price Item", "Misc", 0.0, datetime(2026, 8, 20)),
        (3, "Negative Price Item", "Misc", -5.0, datetime(2026, 8, 20)),
    ]
    columns = ["product_id", "product_name", "category", "unit_price", "_ingested_at"]
    df = spark.createDataFrame(data, columns)

    result = clean_products(df).collect()

    assert len(result) == 1
    assert result[0]["product_id"] == 1


def test_clean_orders_quarantines_invalid_references(spark):
    orders_data = [
        (101, 1, 1, 2, "2026-08-20", datetime(2026, 8, 20)),   # valid
        (102, 999, 1, 1, "2026-08-20", datetime(2026, 8, 20)), # customer_id 999 doesn't exist
        (103, 1, 888, 1, "2026-08-20", datetime(2026, 8, 20)), # product_id 888 doesn't exist
        (104, 1, 1, 0, "2026-08-20", datetime(2026, 8, 20)),   # quantity is zero, invalid
    ]
    orders_columns = ["order_id", "customer_id", "product_id", "quantity", "order_date", "_ingested_at"]
    df_orders = spark.createDataFrame(orders_data, orders_columns)

    df_customers = spark.createDataFrame([(1, "Alex", "Smith")], ["customer_id", "first_name", "last_name"])
    df_products = spark.createDataFrame([(1, "Widget", 9.99)], ["product_id", "product_name", "unit_price"])

    clean, quarantined = clean_orders(df_orders, df_customers, df_products)

    clean_ids = {row["order_id"] for row in clean.collect()}
    quarantined_ids = {row["order_id"] for row in quarantined.collect()}

    assert clean_ids == {101}
    assert quarantined_ids == {102, 103, 104}


def test_clean_orders_computes_order_amount(spark):
    orders_data = [(101, 1, 1, 3, "2026-08-20", datetime(2026, 8, 20))]
    orders_columns = ["order_id", "customer_id", "product_id", "quantity", "order_date", "_ingested_at"]
    df_orders = spark.createDataFrame(orders_data, orders_columns)

    df_customers = spark.createDataFrame([(1, "Alex", "Smith")], ["customer_id", "first_name", "last_name"])
    df_products = spark.createDataFrame([(1, "Widget", 10.00)], ["product_id", "product_name", "unit_price"])

    clean, _ = clean_orders(df_orders, df_customers, df_products)
    result = clean.collect()[0]

    assert result["order_amount"] == 30.00  # 3 * 10.00
