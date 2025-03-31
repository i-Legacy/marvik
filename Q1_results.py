import os
import sys
import argparse
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import col, sum as _sum, count, lag, round
from typing import List
import matplotlib.pyplot as plt


def clean_data(df: DataFrame) -> DataFrame:
    df = df.dropna(
        subset=["created_at_year", "created_at_month", "sale_price", "product_cost"]
    )
    df = df.filter((col("sale_price") >= 0) & (col("product_cost") >= 0))
    df = df.dropDuplicates()

    return df


def read_data_from_s3(
    spark: SparkSession, bucket_name: str, years: List[int]
) -> DataFrame:
    base_path = f"s3a://{bucket_name}/"
    df = spark.read.parquet(base_path)
    df = df.filter(
        (col("created_at_year").isin(years)) & (col("created_at_month").between(1, 3))
    )
    # Print the number of rows grouped by created_at_year
    row_counts = df.groupBy("created_at_year").count()
    row_counts.show()

    df = clean_data(df)

    invalid_data = df.filter(
        (~col("created_at_year").isin(years)) | (~col("created_at_month").between(1, 3))
    )
    if invalid_data.count() > 0:
        raise ValueError(
            "Process contains data outside Q1 or years not included in the years list."
        )

    return df


def transform_data(df: DataFrame) -> DataFrame:
    # Group by year and traffic source to calculate total metrics
    df = df.groupBy("created_at_year", "traffic_source").agg(
        round(_sum(col("sale_price") - col("product_cost")), 2).alias("profit"),
        count("product_id").alias("number_of_purchases"),
    )

    # Define a window partitioned by traffic_source and ordered by year
    window_spec = Window.partitionBy("traffic_source").orderBy("created_at_year")

    # Calculate percentage change in purchases and profit
    df = df.withColumn(
        "percentage_change_purchases",
        (col("number_of_purchases") - lag("number_of_purchases").over(window_spec))
        / lag("number_of_purchases").over(window_spec)
        * 100,
    ).withColumn(
        "percentage_change_profit",
        (col("profit") - lag("profit").over(window_spec))
        / lag("profit").over(window_spec)
        * 100,
    )
    # Round percentage change columns to 2 decimals
    df = df.withColumn(
        "percentage_change_purchases", round(col("percentage_change_purchases"), 2)
    ).withColumn("percentage_change_profit", round(col("percentage_change_profit"), 2))

    # Handle null values for the first row in each partition
    df = df.fillna(
        0, subset=["percentage_change_purchases", "percentage_change_profit"]
    )

    return df


def plot_data(df: DataFrame):
    # Convert Spark DataFrame to Pandas DataFrame for plotting
    pandas_df = df.toPandas()

    # Create a plot
    plt.figure(figsize=(10, 6))
    for traffic_source in pandas_df["traffic_source"].unique():
        source_data = pandas_df[pandas_df["traffic_source"] == traffic_source]
        plt.plot(
            source_data["created_at_year"],
            source_data["profit"],
            marker="o",
            label=traffic_source,
        )

    # Add labels, title, and legend
    plt.title("Profit by Year and Traffic Source")
    plt.xlabel("Year")
    plt.ylabel("Profit")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def main(start_year: int, end_year: int):
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("Read Public S3 Parquet")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
        )
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Define bucket name
    bucket_name = "da-exercise/thelook_ecommerce/purchases/"

    # Generate the list of years
    years = list(range(start_year, end_year + 1))

    # Read and process the data
    data_raw = read_data_from_s3(spark, bucket_name, years)
    data = transform_data(data_raw)

    # Drop duplicates
    data_cleaned = data.dropDuplicates()

    # Export the cleaned data to CSV
    output_path = f"report_{start_year}_{end_year}.csv"
    data_cleaned.write.mode("overwrite").csv(output_path, header=True)
    print(f"Cleaned data saved to {output_path}")

    # Generate and display the plot
    plot_data(data_cleaned)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process data for a given year range.")
    parser.add_argument("from_year", type=int, help="Start year (e.g., 2019)")
    parser.add_argument("to_year", type=int, help="End year (e.g., 2024)")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.from_year, args.to_year)
