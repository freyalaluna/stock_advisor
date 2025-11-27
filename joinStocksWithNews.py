import pyspark.sql.functions as sf
from pyspark.sql import SparkSession, Window

spark = (SparkSession.builder
         .appName("FilterStocks")
         .master("local")
         .getOrCreate())

#Read in our files
allStocks = spark.read.format("parquet").load("/StockAdvisor/datasets/filtered/allStockHistory")
allHeadlines = spark.read.format("parquet").load("/StockAdvisor/datasets/filtered/magSevenHeadlines")

#Date type conversion on the headline DataFrame to make it consistent with the stock DataFrame
allHeadlines = (allHeadlines.select('*', sf.to_date(allHeadlines.timestamp))
                            .drop("timestamp", "headline", "id")
                            .withColumnRenamed("to_date(timestamp)", "date"))

#Left join stocks with headlines based on window starting date
joined = (allStocks.join(allHeadlines, on=[allStocks.ticker == allHeadlines.ticker, allStocks.end_date == allHeadlines.date], how="left")
                   .drop(allHeadlines.ticker)
                   .drop("date"))

#Squash all duplicate rows, aggregating the headline scores that occurred on the same day
squashed = (joined.groupBy(["ticker", "end_date"]).agg(sf.first("start_date").alias("start_date"),
                                                        sf.first("window").alias("window"),
                                                        sf.first("seven_day_target").alias("seven_day_target"),
                                                        sf.collect_list("sentiment_label").alias("sentiment_label"),
                                                        sf.collect_list("sentiment_score").alias("sentiment_score"),))

#Run a window over each ticker, aggregating the last 5 days/10 pieces of news within the partition as a new column
timeframe = Window.partitionBy("ticker").orderBy("end_date").rowsBetween(-5,Window.currentRow)
stocksWithHeadlines = squashed.withColumns({"last_five_sentiments": sf.flatten(sf.collect_list(squashed.sentiment_label).over(timeframe)),
                                             "last_five_score": sf.flatten(sf.collect_list(squashed.sentiment_score).over(timeframe))})

stocksWithHeadlines = (stocksWithHeadlines.withColumns({"last_five_day_sentiments": sf.slice(stocksWithHeadlines.last_five_sentiments, 1, 10),
                                                       "last_five_day_scores": sf.slice(stocksWithHeadlines.last_five_score, 1, 10)})
                                          .drop("sentiment_label", "sentiment_score"))

stocksWithHeadlines = stocksWithHeadlines.select(["ticker", "start_date", "end_date", "window", "last_five_day_sentiments", "last_five_day_scores"])
stocksWithHeadlines.filter(sf.size(stocksWithHeadlines.last_five_day_sentiments) > 0).show(100)
stocksWithHeadlines.write.parquet("/StockAdvisor/datasets/filtered/stocksWithSentiments", mode="overwrite")