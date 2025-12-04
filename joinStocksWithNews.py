import pyspark.sql.functions as sf
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import DoubleType

def execute():

    spark = SparkSession.builder.getOrCreate()

    #Read in our files
    allStocks = spark.read.format("parquet").load("/StockAdvisor/datasets/filtered/allStockHistory")
    allHeadlines = spark.read.format("parquet").load("/StockAdvisor/datasets/filtered/magSevenHeadlines")

    #Date type conversion on the headline DataFrame to make it consistent with the stock DataFrame
    allHeadlines = (allHeadlines.select('*', sf.to_date(allHeadlines.timestamp))
                    .drop("timestamp", "headline", "id")
                    .withColumnRenamed("to_date(timestamp)", "date"))

    allHeadlines = (allHeadlines.withColumn("conf_double", allHeadlines["sentiment_score"].cast(DoubleType()))
                    .drop("sentiment_score")
                    .withColumnRenamed("conf_double", "sentiment_score"))
    #Left join stocks with headlines based on window starting date
    joined = (allStocks.join(allHeadlines, on=[allStocks.ticker == allHeadlines.ticker, allStocks.end_date == allHeadlines.date], how="left")
              .drop(allHeadlines.ticker)
              .drop("date"))

    #Squash all duplicate rows, aggregating the headline scores that occurred on the same day
    squashed = (joined.groupBy(["ticker", "end_date"]).agg(sf.first("start_date").alias("start_date"),
                                                           sf.first("window").alias("stock_highs_30d"),
                                                           sf.first("variance").alias("stock_var"),
                                                           sf.first("seven_day_target").alias("target_price_7d"),
                                                           sf.collect_list("sentiment_label").alias("sentiment_label"),
                                                           sf.collect_list("sentiment_score").alias("sentiment_score"),))

    #Run a window over each ticker, aggregating the last 5 days/10 pieces of news within the partition as a new column
    timeframe = Window.partitionBy("ticker").orderBy("end_date").rowsBetween(-5,Window.currentRow)
    stocksWithHeadlines = squashed.withColumns({"last_five_sentiments": sf.flatten(sf.collect_list(squashed.sentiment_label).over(timeframe)),
                                                "last_five_score": sf.flatten(sf.collect_list(squashed.sentiment_score).over(timeframe))})

    stocksWithHeadlines = (stocksWithHeadlines.withColumns({"news_sentiment_5d": sf.slice(stocksWithHeadlines.last_five_sentiments, 1, 10),
                                                            "news_confidence_5d": sf.slice(stocksWithHeadlines.last_five_score, 1, 10)})
                           .drop("sentiment_label", "sentiment_score"))

    stocksWithHeadlines = stocksWithHeadlines.select(["ticker", "start_date", "end_date", "stock_highs_30d", "target_price_7d",
                                                      "stock_var", "news_sentiment_5d", "news_confidence_5d"])
    print(stocksWithHeadlines.dtypes)
    stocksWithHeadlines.write.parquet("/StockAdvisor/datasets/filtered/stocksWithSentiments", mode="overwrite")