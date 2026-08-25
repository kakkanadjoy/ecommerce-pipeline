# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Ingestion
# MAGIC Reads raw files from the volume `/Volumes/dev_ecommerce/bronze/raw_files/`
# MAGIC and writes them into bronze Delta tables.
# MAGIC
# MAGIC - `customers` and `products` are small reference files: simple batch read + overwrite.
# MAGIC - `orders` is the incremental fact data: read with **Auto Loader**, which tracks
# MAGIC   (via a checkpoint) which files it has already processed, so re-running this
# MAGIC   notebook only picks up NEW order files, not ones already ingested.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, DateType

CATALOG = "dev_ecommerce"
VOLUME_PATH = f"/Volumes/{CATALOG}/bronze/raw_files"
CHECKPOINT_PATH = f"/Volumes/{CATALOG}/bronze/raw_files/_checkpoints/orders"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest customers (batch, reference data)

# COMMAND ----------

customers_schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("signup_date", DateType(), True),
    StructField("region", StringType(), True),
])

df_customers = (
    spark.read
    .option("header", True)
    .schema(customers_schema)
    .csv(f"{VOLUME_PATH}/customers.csv")
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

(
    df_customers.write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(f"{CATALOG}.bronze.customers")
)

print(f"Ingested {df_customers.count()} customer rows into {CATALOG}.bronze.customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingest products (batch, reference data)

# COMMAND ----------

products_schema = StructType([
    StructField("product_id", IntegerType(), False),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("unit_price", DoubleType(), True),
])

df_products = (
    spark.read
    .option("header", True)
    .schema(products_schema)
    .csv(f"{VOLUME_PATH}/products.csv")
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

(
    df_products.write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(f"{CATALOG}.bronze.products")
)

print(f"Ingested {df_products.count()} product rows into {CATALOG}.bronze.products")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ingest orders (incremental, via Auto Loader)
# MAGIC This is the important one: Auto Loader tracks which files it has already
# MAGIC read using the checkpoint location. Every time this cell runs, it only
# MAGIC processes files it hasn't seen before — this is what "incremental ingestion"
# MAGIC actually looks like in practice, not a full reload every time.

# COMMAND ----------

orders_schema = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("customer_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("order_date", DateType(), True),
    StructField("order_ts", TimestampType(), True),
])

df_orders_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", True)
    .schema(orders_schema)
    .load(f"{VOLUME_PATH}/")
    .filter(F.col("_metadata.file_path").contains("orders_"))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

query = (
    df_orders_stream.writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(availableNow=True)   # process everything currently available, then stop (not a 24/7 stream)
    .toTable(f"{CATALOG}.bronze.orders")
)

query.awaitTermination()

print(f"Bronze orders table row count: {spark.table(f'{CATALOG}.bronze.orders').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Quick sanity check

# COMMAND ----------

display(spark.table(f"{CATALOG}.bronze.orders").orderBy(F.desc("_ingested_at")).limit(10))