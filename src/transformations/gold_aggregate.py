"""
Gold layer aggregation for the ecommerce pipeline.

Same design principle as silver: pure aggregation functions (DataFrame in,
DataFrame out) are separated from I/O (reading silver tables, writing gold
tables). The pure functions are unit tested in tests/test_gold_aggregate.py
with no Databricks dependency at all.

Unlike silver (which uses incremental MERGE), gold tables here are fully
recomputed and overwritten on each run. This matches how gold is typically
handled in practice: aggregates are cheap enough, and small enough, to just
rebuild from silver each time, rather than maintaining incremental delta
logic for every summary table. (A very large enterprise might instead
recompute only a rolling window — e.g. the last 90 days — but a full
overwrite is the simpler and more common starting point.)
"""

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger("gold_aggregate")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Pure aggregation functions — no I/O, no Databricks dependency.
# ---------------------------------------------------------------------------

def build_daily_sales_summary(df_silver_orders: DataFrame, df_silver_products: DataFrame) -> DataFrame:
    """
    Aggregates orders by order_date and product category:
    total orders, total units sold, and total revenue per day per category.
    This is the classic "time-series" gold pattern — built for a dashboard
    showing sales trends over time.
    """
    enriched = df_silver_orders.join(
        df_silver_products.select("product_id", "category"), on="product_id", how="inner"
    )

    return (
        enriched.groupBy("order_date", "category")
        .agg(
            F.countDistinct("order_id").alias("total_orders"),
            F.sum("quantity").alias("total_units_sold"),
            F.round(F.sum("order_amount"), 2).alias("total_revenue"),
        )
        .orderBy("order_date", "category")
    )


def build_customer_lifetime_value(df_silver_orders: DataFrame, df_silver_customers: DataFrame) -> DataFrame:
    """
    Aggregates orders by customer: total number of orders, total amount
    spent, and first/last order dates. This is the classic "entity-level"
    gold pattern — built for a customer-analytics or CRM-facing dashboard.
    """
    per_customer = df_silver_orders.groupBy("customer_id").agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.round(F.sum("order_amount"), 2).alias("lifetime_value"),
        F.min("order_date").alias("first_order_date"),
        F.max("order_date").alias("last_order_date"),
    )

    return per_customer.join(
        df_silver_customers.select("customer_id", "first_name", "last_name", "region"),
        on="customer_id",
        how="left",
    ).orderBy(F.desc("lifetime_value"))


# ---------------------------------------------------------------------------
# I/O orchestration — reads silver, calls the pure functions, overwrites gold.
# ---------------------------------------------------------------------------

def run_gold_pipeline(spark: SparkSession, catalog: str = "dev_ecommerce") -> None:
    logger.info("Starting gold pipeline for catalog=%s", catalog)

    silver_orders = spark.table(f"{catalog}.silver.orders")
    silver_products = spark.table(f"{catalog}.silver.products")
    silver_customers = spark.table(f"{catalog}.silver.customers")

    daily_sales = build_daily_sales_summary(silver_orders, silver_products)
    customer_ltv = build_customer_lifetime_value(silver_orders, silver_customers)

    (
        daily_sales.write.mode("overwrite")
        .format("delta")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.gold.daily_sales_summary")
    )
    (
        customer_ltv.write.mode("overwrite")
        .format("delta")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.gold.customer_lifetime_value")
    )

    logger.info(
        "Gold pipeline complete. daily_sales_summary rows=%s, customer_lifetime_value rows=%s",
        daily_sales.count(), customer_ltv.count(),
    )


if __name__ == "__main__":
    spark_session = SparkSession.builder.getOrCreate()
    run_gold_pipeline(spark_session)
