"""
Unit tests for gold_aggregate.py — pure aggregation logic, tested with a
local Spark session, no Databricks dependency.

Run locally with: pytest tests/test_gold_aggregate.py -v
"""

import pytest
from pyspark.sql import SparkSession

from src.transformations.gold_aggregate import (
    build_daily_sales_summary,
    build_customer_lifetime_value,
)


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("gold_aggregate_tests")
        .getOrCreate()
    )


def test_daily_sales_summary_aggregates_correctly(spark):
    orders_data = [
        (1, 1, 10, 2, "2026-08-20", 20.00),  # order_id, customer_id, product_id, quantity, order_date, order_amount
        (2, 2, 10, 1, "2026-08-20", 10.00),
        (3, 1, 20, 3, "2026-08-21", 45.00),
    ]
    orders_columns = ["order_id", "customer_id", "product_id", "quantity", "order_date", "order_amount"]
    df_orders = spark.createDataFrame(orders_data, orders_columns)

    products_data = [(10, "Widget", "Electronics", 10.00), (20, "Gadget", "Electronics", 15.00)]
    df_products = spark.createDataFrame(products_data, ["product_id", "product_name", "category", "unit_price"])

    result = build_daily_sales_summary(df_orders, df_products).collect()

    day1 = [r for r in result if r["order_date"] == "2026-08-20"][0]
    assert day1["total_orders"] == 2
    assert day1["total_units_sold"] == 3
    assert day1["total_revenue"] == 30.00

    day2 = [r for r in result if r["order_date"] == "2026-08-21"][0]
    assert day2["total_orders"] == 1
    assert day2["total_units_sold"] == 3
    assert day2["total_revenue"] == 45.00


def test_customer_lifetime_value_aggregates_correctly(spark):
    orders_data = [
        (1, 1, 10, 2, "2026-08-20", 20.00),
        (2, 1, 20, 1, "2026-08-22", 15.00),
        (3, 2, 10, 1, "2026-08-21", 10.00),
    ]
    orders_columns = ["order_id", "customer_id", "product_id", "quantity", "order_date", "order_amount"]
    df_orders = spark.createDataFrame(orders_data, orders_columns)

    customers_data = [(1, "Alex", "Smith", "US-East"), (2, "Priya", "Patel", "APAC")]
    df_customers = spark.createDataFrame(customers_data, ["customer_id", "first_name", "last_name", "region"])

    result = build_customer_lifetime_value(df_orders, df_customers).collect()

    cust1 = [r for r in result if r["customer_id"] == 1][0]
    assert cust1["total_orders"] == 2
    assert cust1["lifetime_value"] == 35.00
    assert cust1["first_order_date"] == "2026-08-20"
    assert cust1["last_order_date"] == "2026-08-22"
    assert cust1["region"] == "US-East"

    cust2 = [r for r in result if r["customer_id"] == 2][0]
    assert cust2["total_orders"] == 1
    assert cust2["lifetime_value"] == 10.00
