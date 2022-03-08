from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession, dataframe
from pyspark.sql import SQLContext
from utils import TextProc, Bert, PreProc
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
    # file_path = "/Users/raviguntur/Nautilus/DroneVision/news_clustering/data/news_dataset.csv"

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

    # 2. get the dimensionality for categories
    category_dict = PreProc.get_categories(df, col='category')

    # 3. initialize bert model and TextProc
    bert = Bert()
    text_proc = TextProc()

    # 4. convert specific columns into their embeddings and drop columns not being used
    df = df.select('ID', 'category', 'popularity_score', 'headline', 'short_description')
    print("number of datapoints", df.count())

    # 5. process categories, popularity_score fields, and text fields into vectorized form
    df = PreProc.categories_to_onehot(df, 'category', category_dict)
    df = PreProc.min_max(df, 'popularity_score')
    df_with_embedding = text_proc.text_to_embeddings(df,
                                                     bert.get_tokenizer(),
                                                     bert.get_model())

    # 6. drop raw text columns and keep only the embeddings columns
    df_embeddings = df_with_embedding.select('ID', 'category', 'category_one_hot', 'normalized_popularity_score', 'headline_embedding',
                                             'short_description_embedding').persist(StorageLevel.DISK_ONLY_2)

    # 7. compute the nearest neighbour matches
    # dist_calculated_df = text_proc.distance_computation_v2(df_embeddings).persist(StorageLevel.DISK_ONLY_2)
    # print("number of datapoints", dist_calculated_df.count())

    dist_calculated_df = text_proc.distance_computation_v3(sql_context, df_embeddings).persist(StorageLevel.DISK_ONLY_2)

    # 8. store the ID field, and the corresponding nearest neighbours and scores in a parquet file
    dist_calculated_df.write.parquet('nearest_neighbour_full_vector')

