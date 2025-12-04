from datetime import timedelta
from functools import reduce

import pyspark.sql.functions as sf
from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType


def createStockDataframe(stock_path, file_schema):
    stock_name = stock_path.split("/")[-1].split(".")[0]

    spark = SparkSession.builder.getOrCreate()

    stock_history = spark.read.csv(stock_path, schema=file_schema, header=True)
    stock_history = (stock_history.withColumn("date", sf.to_date(stock_history.date, "yyyy-MM-dd"))
                     .withColumn("ticker", sf.lit(stock_name))
                     .drop("open", "close", "adjClose", "volume", "low"))

    min_max_timestamps = stock_history.agg(sf.min(stock_history.date), sf.max(stock_history.date)).head().asDict()

    first_date = min_max_timestamps["min(date)"]
    last_date = min_max_timestamps["max(date)"]
    all_dates = [first_date + timedelta(days=d)
                 for d in range((last_date - first_date).days + 1)]
    stock_timeframe = (spark.createDataFrame(all_dates, DateType())
                       .withColumnRenamed("value", "date"))
    total_stock_history = ((stock_history.join(stock_timeframe, stock_history.date == stock_timeframe.date, "right")
                            .fillna(value=0, subset="high"))
                           .fillna(value=stock_name, subset="ticker")).drop(stock_history.date)

    return total_stock_history

def execute(files):
    spark = SparkSession.builder.getOrCreate()

    # Read in stock file argument from HDFS into DataFrame
    # with columns: Date,Open,High,Low,Close,AdjustedClose,Volume
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("adjClose", DoubleType(), True),
        StructField("volume", IntegerType(), True),
    ])


    histories = [createStockDataframe(s, schema) for s in files]
    totalStockHistory = reduce(DataFrame.unionByName, histories)

    #Create a sliding window that aggregates the next 30 days of highs starting from the current row,
    #and collect the target value seven days out from the end
    timeframe = Window.partitionBy("ticker").orderBy("date").rowsBetween(Window.currentRow, 30)
    seven_after = Window.partitionBy("ticker").orderBy("date")
    allStockWindows = totalStockHistory.withColumns({"end_date": sf.date_add("date", 30),
                                                     "window": sf.collect_list(totalStockHistory.high).over(timeframe),
                                                     "variance": sf.variance(totalStockHistory.high).over(timeframe),
                                                     "seven_day_target": sf.lead(totalStockHistory.high, 37).over(seven_after)
                                                     })

    allStockWindows = allStockWindows.withColumnRenamed("date", "start_date").drop("high")
    allStockWindows = allStockWindows.filter(allStockWindows.seven_day_target != 0.0)
    allStockWindows.write.parquet("/StockAdvisor/datasets/filtered/allStockHistory", mode="overwrite")