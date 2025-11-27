from pyspark import SparkFiles
from transformers import pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, MapType

spark = (SparkSession.builder
            .appName("FilterNews")
            .master("yarn")
            .getOrCreate())

classifier = pipeline('text-classification', model='tabularisai/ModernFinBERT')

@udf (returnType=MapType(StringType(), StringType()))
def classify_sentiment(headline):
    sentiment = classifier(headline)
    return sentiment[0]

targetStocks = ["GOOGL", "AAPL", "FB", "NVDA", "AMZN", "MSFT"]

schema = StructType([
    StructField("index", StringType(), True),
    StructField("headline", StringType(), True),
    StructField("url", StringType(), True),
    StructField("publisher", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("ticker", StringType(), True),
])

#index, headline, URL, article author/publisher, publication timestamp, stock ticker symbol
analystHeadlinesDataFrame = spark.read.csv("/StockAdvisor/datasets/FinancialNews/raw_analyst_ratings.csv", schema=schema, header=True)
partnerHeadlinesDataFrame = spark.read.csv("/StockAdvisor/datasets/FinancialNews/raw_partner_headlines.csv", schema=schema, header=True)

#Filter out all non-mag7 stocks
magSevenAnalystHeadlines = analystHeadlinesDataFrame.filter(analystHeadlinesDataFrame.ticker.isin(targetStocks))
magSevenPartnerHeadlines = partnerHeadlinesDataFrame.filter(partnerHeadlinesDataFrame.ticker.isin(targetStocks))

#Join both tables
magSevenHeadlines = magSevenAnalystHeadlines.union(magSevenPartnerHeadlines)
magSevenHeadlines = magSevenHeadlines.withColumn("timestamp", sf.to_timestamp("timestamp"))

#Sort headlines by ticker and date
magSevenHeadlinesSorted = magSevenHeadlines.orderBy("ticker", sf.asc("timestamp"))

#Do BERT stuff for each headline, append sentiment to end of each articleinfo list
# <result sentiment, confidence>
magSevenHeadlinesSorted = magSevenHeadlinesSorted.withColumn("sentiment", classify_sentiment(magSevenHeadlinesSorted.headline))
magSevenHeadlinesSorted = magSevenHeadlinesSorted.withColumns({"sentiment_label": magSevenHeadlinesSorted.sentiment.label, "sentiment_score": magSevenHeadlinesSorted.sentiment.score}) \
                          .drop("sentiment", "url", "publisher", "index")
magSevenHeadlinesSorted = magSevenHeadlinesSorted.withColumn("id", monotonically_increasing_id())

# <ticker, [list of [id, publish date, sentiment, sentiment confidence]]>
magSevenHeadlinesSorted.write.parquet("/StockAdvisor/datasets/filtered/magSevenHeadlines", mode="overwrite")