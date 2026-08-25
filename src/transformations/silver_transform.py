"""
Silver layer transformation for the ecommerce pipeline.

Design principle (this is the important part): transformation LOGIC (pure
functions that take DataFrames in and return DataFrames out) is kept
completely separate from I/O (reading bronze tables, writing to Delta via
MERGE). This is what makes the logic unit-testable without a live Databricks
cluster, a real catalog, or real tables — tests exercise the pure functions
directly with small sample DataFrames. Only `run_silver_pipeline` touches
real tables, and that function is what actually runs inside Databricks.

Enterprise patterns demonstrated here:
  - Deduplication using the most recently ingested row per business key
  - Referential integrity checks (orders must reference real customers/products)
  - A "quarantine" pattern: rows that fail validation are NOT silently
    dropped — they're written to a separate table so someone can investigate
  - Idempotent writes via MERGE (safe to re-run without creating duplicates)
  - Logging at each stage, instead of silent success/failure
"""

import logging
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger("silver_transform")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Pure transformation functions.
# No spark.table(), no .write, no catalog references — just DataFrame in,
# DataFrame out. This is what tests/test_silver_transform.py exercises.
# ---------------------------------------------------------------------------

def clean_customers(df_bronze_customers: DataFrame) -> DataFrame:
    """
    Deduplicate customers on customer_id, keeping the most recently ingested
    row per id. Trims whitespace on name fields. Drops rows with a null id.
    """
    df = df_bronze_customers.filter(F.col("customer_id").isNotNull())

    window = Window.partitionBy("customer_id").orderBy(F.col("_ingested_at").desc())
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
        .withColumn("first_name", F.trim(F.col("first_name")))
        .withColumn("last_name", F.trim(F.col("last_name")))
    )


def clean_products(df_bronze_products: DataFrame) -> DataFrame:
    """
    Deduplicate products on product_id, keeping the most recently ingested
    row. Drops any product with a non-positive unit_price — a basic data
    quality rule; a negative or zero price is treated as bad source data.
    """
    df = df_bronze_products.filter(
        F.col("product_id").isNotNull() & (F.col("unit_price") > 0)
    )
    window = Window.partitionBy("product_id").orderBy(F.col("_ingested_at").desc())
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


def clean_orders(
    df_bronze_orders: DataFrame,
    df_clean_customers: DataFrame,
    df_clean_products: DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """
    Deduplicates orders on order_id (latest by _ingested_at wins), enforces
    referential integrity against clean customers/products, drops orders
    with non-positive quantity, and computes order_amount.

    Returns (clean_orders_df, quarantined_orders_df). Invalid rows are
    returned, not dropped silently — the caller decides what to do with them
    (in run_silver_pipeline, they get written to a quarantine table).
    """
    window = Window.partitionBy("order_id").orderBy(F.col("_ingested_at").desc())
    deduped = (
        df_bronze_orders.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )

    # leftsemi keeps only rows from `deduped` that have a matching key in the
    # dimension table — this is the referential integrity check.
    valid_refs = (
        deduped
        .join(df_clean_customers.select("customer_id"), on="customer_id", how="leftsemi")
        .join(df_clean_products.select("product_id"), on="product_id", how="leftsemi")
    )
    clean = valid_refs.filter(F.col("quantity") > 0)

    # leftanti gives back everything in `deduped` NOT present in `clean` —
    # i.e. exactly the rows that failed a validation rule above.
    quarantined = deduped.join(clean.select("order_id"), on="order_id", how="leftanti")

    clean_with_amount = (
        clean.join(
            df_clean_products.select("product_id", "unit_price"),
            on="product_id",
            how="inner",
        )
        .withColumn("order_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
    )

    return clean_with_amount, quarantined


# ---------------------------------------------------------------------------
# I/O orchestration. This is what actually runs in Databricks. Not unit
# tested directly (it needs a real cluster + Unity Catalog) — this is what
# an integration test or manual notebook run validates instead.
# ---------------------------------------------------------------------------

def _merge_into_table(spark: SparkSession, df: DataFrame, table_name: str, key_column: str) -> None:
    """Upserts df into table_name on key_column; creates the table on first run."""
    from delta.tables import DeltaTable

    if not spark.catalog.tableExists(table_name):
        df.write.format("delta").saveAsTable(table_name)
        logger.info("Created new silver table %s", table_name)
        return

    delta_table = DeltaTable.forName(spark, table_name)
    (
        delta_table.alias("target")
        .merge(df.alias("source"), f"target.{key_column} = source.{key_column}")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info("Merged into %s", table_name)


def run_silver_pipeline(spark: SparkSession, catalog: str = "dev_ecommerce") -> None:
    """Reads bronze, cleans it, and upserts the result into silver tables."""
    logger.info("Starting silver pipeline for catalog=%s", catalog)

    bronze_customers = spark.table(f"{catalog}.bronze.customers")
    bronze_products = spark.table(f"{catalog}.bronze.products")
    bronze_orders = spark.table(f"{catalog}.bronze.orders")

    clean_cust = clean_customers(bronze_customers)
    clean_prod = clean_products(bronze_products)
    clean_ord, quarantined_ord = clean_orders(bronze_orders, clean_cust, clean_prod)

    quarantine_count = quarantined_ord.count()
    if quarantine_count > 0:
        logger.warning("%s order rows quarantined (failed validation) this run", quarantine_count)
        (
            quarantined_ord.write.mode("append")
            .format("delta")
            .saveAsTable(f"{catalog}.silver.orders_quarantine")
        )

    _merge_into_table(spark, clean_cust, f"{catalog}.silver.customers", "customer_id")
    _merge_into_table(spark, clean_prod, f"{catalog}.silver.products", "product_id")
    _merge_into_table(spark, clean_ord, f"{catalog}.silver.orders", "order_id")

    logger.info(
        "Silver pipeline complete. customers=%s products=%s orders=%s quarantined=%s",
        clean_cust.count(), clean_prod.count(), clean_ord.count(), quarantine_count,
    )


if __name__ == "__main__":
    spark_session = SparkSession.builder.getOrCreate()
    run_silver_pipeline(spark_session)
