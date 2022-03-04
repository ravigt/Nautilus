from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession, dataframe
from pyspark.sql import SQLContext
from utils import TextProc, Bert


def load_data(sqc, file_path: str) -> dataframe:
    df = sqc.read.csv(
        file_path,
        header=True,
        mode="DROPMALFORMED",
    )
    return df


if __name__ == "__main__":
    # file_path = "small_data_10k.csv"
    file_path = "news_dataset.csv"

    conf = SparkConf()
    conf.set("spark.executor.heartbeatInterval", "80000s")
    conf.set("spark.network.timeout", "80001s")
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

    # 1. load data from csv
    df = load_data(sql_context, file_path)

    # 2. initialize bert model and TextProc
    bert = Bert()
    text_proc = TextProc()

    # 3. convert specific columns into their embeddings and drop columns not being used
    df = df.select('ID', 'category', 'headline', 'short_description').repartition(100).persist(StorageLevel.DISK_ONLY_2)
    print("number of datapoints", df.count())

    df_with_embedding = text_proc.text_to_embeddings(df,
                                                     bert.get_tokenizer(),
                                                     bert.get_model()).repartition(100)

    # 4. drop raw text columns and keep only the embeddings columns
    df_embeddings = df_with_embedding.select('ID', 'category', 'headline_embedding', 'short_description_embedding')

    # 5. compute the nearest neighbour matches
    dist_calculated_df = text_proc.distance_computation_v2(df_embeddings).persist(StorageLevel.DISK_ONLY_2)
    print("number of datapoints", dist_calculated_df.count())

    # 6. store the ID field, and the corresponding nearest neighbours and scores in a parquet file
    dist_calculated_df.write.parquet('nearest_neighbour_parquet_full')

