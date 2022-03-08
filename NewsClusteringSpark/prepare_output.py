import logging
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, dataframe
from pyspark.sql import SQLContext
from utils import TextProc
from pyspark.sql import functions as F


def load_data(sqc, file_path: str) -> dataframe:
    df = sqc.read.csv(
        file_path,
        header=True,
        mode="DROPMALFORMED",
    )
    return df


if __name__ == "__main__":
    file_path = "/Users/raviguntur/Nautilus/DroneVision/news_clustering/data/small_data_10k.csv"
    file_path_nearest_neighbours = "/Users/raviguntur/Nautilus/DroneVision/news_clustering/nearest_neighbour_full_vector"

    conf = SparkConf()
    conf.set("spark.executor.heartbeatInterval", "8000s")
    conf.set("spark.network.timeout", "8001s")
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    conf.set("spark.sql.dynamicPartitionPruning.enabled", "true")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "12g")
    conf.set("spark.driver.cores", "1")
    conf.set("spark.executor.cores", "2")
    conf.set("spark.driver.maxResultSize", "10g")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.sql.broadcastTimeout", "4000")
    conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    sc = SparkContext(conf=conf)
    spark_session = SparkSession.builder \
        .config(conf=sc.getConf()) \
        .getOrCreate()
    sql_context = SQLContext(sc, spark_session)

    df = load_data(sql_context, file_path)
    df_nn = sql_context.read.parquet(file_path_nearest_neighbours)

    df.join(df_nn, on='ID') \
    .withColumnRenamed('headline', 'Headline') \
    .withColumnRenamed('short_description', "Short_Description") \
    .withColumnRenamed('category', "Category") \
    .withColumnRenamed('date', "Date") \
    .withColumnRenamed('popularity_score', "Popularity_Score") \
    .withColumnRenamed('nearest_neighbour', "Similar_News_Items_Comma_separated") \
    .withColumnRenamed('scores', "Similarity_Scores_Comma_separated").toPandas().to_csv('result_data_10k.csv', sep='|')