import logging
from pyspark.sql import functions as F
from pyspark.sql import types as T
import numpy as np
import torch
import transformers
import re


class Bert:
    def __init__(self):
        """
        initialize a pre-trained distilBERT model
        """
        self.model_class, self.tokenizer_class, self.pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer,
                                                                           'distilbert-base-uncased')
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = self.model_class.from_pretrained(self.pretrained_weights)

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model


class PreProc:
    """
    Pre-processing helper functions
    """
    default_category = "DEFAULT"

    @staticmethod
    def normalize_random_category(text):
        """
        many categories are not real categories. there seems to be noisy category data. this helper function assumes that if all characters are in
        upper case then it is a category label and if there are lower case words then it is noise. The noisy ones are re-labeled as DEFAULT.
        :param text: the text that should be validated
        :return: text or a default value
        """
        # This is a UDF. There are random strings in the category label. normalize them to a default value
        try:
            normalized = text if text.isupper() else PreProc.default_category
        except:
            normalized = PreProc.default_category
        return normalized

    @staticmethod
    def get_categories(df, col):
        """
        we are assuming that a record cannot belong to multiple categories. if it were so, we need the delimiter
        :param df: dataframe
        :param col: the column name that needs to be processed
        :return: dictionary of categories and their index to be used in creating a one-hot vector
        """

        normalize_random_category_udf = F.udf(PreProc.normalize_random_category, T.StringType())
        categories = df.select(col)\
            .withColumn('categories', normalize_random_category_udf(col))\
            .select('categories').distinct().collect()
        categories = sorted(list(map(lambda x: x['categories'], categories)))
        categories_dict = dict(list(zip(categories, range(0, len(categories)))))
        return categories_dict

    @staticmethod
    def categories_to_onehot(df, col, categories_dict):
        """
        adds a new column with the one-hot encoded category
        :param df: dataframe
        :param col: column that needs to be encoded
        :param categories_dict: categories dictionary
        :return: dataframe with a new column of one-hot encoded category
        """
        def one_hot(col):
            vec = np.zeros(len(categories_dict.keys()))
            key = PreProc.normalize_random_category(col)
            if key in categories_dict:
                vec[categories_dict[key]] = 1
            else:
                vec[categories_dict[PreProc.default_category]] = 1
            return vec.tolist()

        udf_one_hot = F.udf(one_hot, T.ArrayType(T.FloatType()))
        df = df.withColumn('category_one_hot', udf_one_hot(col))
        return df

    @staticmethod
    def min_max(df, col):
        """
        performs min-max normalization
        Note: there could be noisy popularity scores. For example score of 20120128 which looks like a date. Perhaps a comma is missing and the
        csv parser has wrongly parsed it. So we will assume that scores are always <= 1
        for cases where the score is very high we will set it to 0 because it most likely is not a score
        +----------------+
          | popularity_score |
          +----------------+
          | 0.370420601 |
          | 0.346832474 |
          | 0.355386286 |
          | 0.381655249 |
          | 0.355655857 |
          | 0.381884497 |
          | 0.349741362 |
          | 20120128 |
          | 0.341751892 |
          +----------------+
        :param df: dataframe
        :param col: column that needs to be normalized
        :return: dataframe with normalized column
        """
        cleaned_df = df\
            .withColumn('cleaned', F.when((F.col(col) <= 1) & (F.col(col) >= 0), F.col(col)).otherwise(0))

        max_val = float(cleaned_df.select(F.max(F.col('cleaned')).alias('min_val')).collect()[0]['min_val'])
        min_val = float(cleaned_df.select(F.min(F.col('cleaned')).alias('max_val')).collect()[0]['max_val'])

        def normalize_min_max(data):
            return (float(data) - min_val)/(max_val - min_val)

        udf_min_max = F.udf(normalize_min_max, T.FloatType())
        cleaned_df = cleaned_df.withColumn('normalized_popularity_score', udf_min_max(F.col('cleaned')))
        print("Min popularity_score: {} max popularity_score:{}".format(min_val, max_val))
        return cleaned_df


class TextProc:
    """
    utility functions to process text data
    """
    def __init__(self):
        pass

    def get_embedding_vec(self, text, tokenizer, model):
        """
        converts an input text to an embedding using the distilbert model.
        :param text: input text
        :return: embedding vector
        """

        # truncate text to this nnumber of words to speed up while testing
        MAX_TEXT = 100

        if text is None:
            text = ""

        # in order to speed-up computation on a non gpu machine with no spark cluster
        # truncating the text to a few words
        words = re.findall("([\w][\w']*\w)", text)
        text = " ".join(words[:MAX_TEXT])

        # get the feature vector form bert
        tokens = tokenizer.encode(text, add_special_tokens=True)
        num_tokens = len(tokens)

        input_ids = torch.tensor(np.array(tokens))
        input_ids = input_ids.reshape((num_tokens, 1))
        with torch.no_grad():
            last_hidden_states = model(input_ids)

        # for every text block word embeddings are generated. the mean of all the embeddings
        # is taken as the embedding for the text block
        feature_vec = torch.mean(last_hidden_states[0][:, 0, :], dim=0).numpy()
        # print(text)
        # print(last_hidden_states[0].shape)
        return feature_vec

    def text_to_embeddings(self, df, tokenizer, model):
        """
        converts the headline and short description columns into their corresponding feature vector representation
        :param df: full spark dataframe
        :return: dataframe with additional columns containing the feature vector form of the text
        """
        def feature_vec(colum):
            vec = self.get_embedding_vec(colum, tokenizer, model)
            return vec.tolist()

        feature_vec_udf = F.udf(feature_vec, T.ArrayType(T.FloatType()))
        df_embed = df\
            .withColumn("headline_embedding", feature_vec_udf(F.col('headline')))\
            .withColumn("short_description_embedding", feature_vec_udf(F.col('short_description')))
        return df_embed

    def distance_computation(self, df, df_other):
        """
        This is a user defined function to compute distance between rows of a dataframe
        Approach: avg(euclidean distance between headlines + euclidean distance between descriptions)
                  given that the category is the same. This means if categories match the distance on category is 1 else 0.
                  This is factored-in within the join statement
        :param df: source df
        :param df_other: other data df (same as the source df as we are performing a crossJoin)
        :return:
        """
        def dist(headline_a, headline_b, desc_a, desc_b):
            hd = np.linalg.norm(np.array(headline_a) - np.array(headline_b))
            dd = np.linalg.norm(np.array(desc_a) - np.array(desc_b))

            # all three get the same weightage
            return float((hd+dd)/2)

        def knn(neighbour_list):
            """
            the group-by on the ID field gives the distances between that row and all rows. From among the distances top-2 are selected
            :param neighbour_list: list of distances for this row
            :return: a tuple of the 2 distances and 2 IDs corresponding to those distances. They are comma separated strings
            """
            NUM_NEAREST = 2
            nearest2 = sorted(neighbour_list, key=lambda a: a[0], reverse=False)[:NUM_NEAREST]
            dist_str = ",".join(list(map(lambda x: str(x[0]), nearest2)))
            id_str = ",".join(list(map(lambda x: str(x[1]), nearest2)))
            return dist_str, id_str

        # udf for computing distances
        udf_dist = F.udf(dist, T.FloatType())
        # udf to get the top 2 closest distances
        udf_knn = F.udf(knn, T.StructType([T.StructField("dist_str",T.StringType()), T.StructField("id_str", T.StringType())]))

        # the data frame processing step involving the join
        df_distance_calculated = df.repartition(10).crossJoin(df_other.repartition(10)) \
            .where((F.col('ID') != F.col('ID_other')) & (F.col('category') == F.col('category_other'))) \
            .withColumn("dist", udf_dist('headline_embedding', 'headline_embedding_other', 'short_description_embedding',
                                         'short_description_embedding_other')) \
            .withColumn("tup", F.struct(F.col('dist'), F.col('ID_other'))) \
            .groupBy("ID") \
            .agg(F.collect_list('tup').alias('dist_against_others')) \
            .withColumn('2neighbour', udf_knn(F.col('dist_against_others'))) \
            .select('ID', F.col('2neighbour.dist_str').alias('scores'), F.col('2neighbour.id_str').alias('nearest_neighbour'))
        return df_distance_calculated

    def distance_computation_parallel(self, df, df_other):
        """
        This is a user defined function to compute distance between rows of a dataframe
        Approach: avg(euclidean distance between headlines + euclidean distance between descriptions)
                  given that the category is the same. This means if categories match the distance on category is 1 else 0.
                  This is factored-in within the join statement
        :param df: source df
        :param df_other: other data df (same as the source df as we are performing a crossJoin)
        :return:
        """
        def dist(headline_a, headline_b, desc_a, desc_b):
            hd = np.linalg.norm(np.array(headline_a) - np.array(headline_b))
            dd = np.linalg.norm(np.array(desc_a) - np.array(desc_b))

            # all three get the same weightage
            return float((hd+dd)/2)

        def knn(neighbour_list):
            """
            the group-by on the ID field gives the distances between that row and all rows. From among the distances top-2 are selected
            :param neighbour_list: list of distances for this row
            :return: a tuple of the 2 distances and 2 IDs corresponding to those distances. They are comma separated strings
            """
            NUM_NEAREST = 2
            nearest2 = sorted(neighbour_list, key=lambda a: a[0], reverse=False)[:NUM_NEAREST]
            dist_str = ",".join(list(map(lambda x: str(x[0]), nearest2)))
            id_str = ",".join(list(map(lambda x: str(x[1]), nearest2)))
            return dist_str, id_str

        # udf for computing distances
        udf_dist = F.udf(dist, T.FloatType())
        # udf to get the top 2 closest distances
        udf_knn = F.udf(knn, T.StructType([T.StructField("dist_str",T.StringType()), T.StructField("id_str", T.StringType())]))

        # the data frame processing step involving the join
        df_distance_calculated = df.repartition(10).crossJoin(df_other.repartition(10)) \
            .where((F.col('ID') != F.col('ID_other')) & (F.col('category') == F.col('category_other'))) \
            .withColumn("dist", udf_dist('headline_embedding', 'headline_embedding_other', 'short_description_embedding',
                                         'short_description_embedding_other')) \
            .withColumn("tup", F.struct(F.col('dist'), F.col('ID_other'))) \
            .groupBy("ID") \
            .agg(F.collect_list('tup').alias('dist_against_others')) \
            .withColumn('2neighbour', udf_knn(F.col('dist_against_others'))) \
            .select('ID', F.col('2neighbour.dist_str').alias('scores'), F.col('2neighbour.id_str').alias('nearest_neighbour'))
        return df_distance_calculated

    def distance_computation_v2(self, df):
        """
        This is a user defined function to compute distance between rows of a dataframe
        Approach: avg(euclidean distance between headlines + euclidean distance between descriptions)
                  given that the category is the same. This means if categories match the distance on category is 1 else 0.
                  This is factored-in within the join statement
        :param df: source df
        :param df_other: other data df (same as the source df as we are performing a crossJoin)
        :return:
        """

        def knn_loops(data):
            """
            implements an O(n^2) comparison between elements within a category.
            better to use distance_computation() if we have a true spark cluster. Current approach is to basically adapt to a purely memory based
            implementation.
            :param data: list of type Row(). These are all items of the same 'category'
            :return: list of items of the form (ID, "score1,score1", "if of neighbour1,id of neighbour2")
            """
            print(len(data))
            score_list = list()
            for src in data:
                top2 = [1000, 1000]
                id2 = ["-1", "-1"]
                for dst in data:
                    if src.ID == dst.ID:
                        continue
                    hd = np.linalg.norm(np.array(src.headline_embedding) - np.array(dst.headline_embedding))
                    dd = np.linalg.norm(np.array(src.short_description_embedding) - np.array(dst.short_description_embedding))
                    dist = (hd + dd) / 2
                    if dist < top2[0]:
                        top2[0] = dist
                        id2[0] = dst.ID
                    elif dist < top2[1]:
                        top2[1] = dist
                        id2[1] = dst.ID

                for idx, i in enumerate(top2):
                    top2[idx] = 0 if top2[idx] == 1000 else top2[idx]
                for idx, i in enumerate(id2):
                    id2[idx] = "" if id2[idx] == "-1" else id2[idx]
                score_list.append((src.ID, "{},{}".format(top2[0], top2[1]), "{},{}".format(id2[0], id2[1])))
            return score_list

        # UDF with schema specified
        udf_knn_loops = F.udf(knn_loops, T.ArrayType(T.StructType([T.StructField("ID", T.StringType()),
                                                                   T.StructField("scores", T.StringType()),
                                                                   T.StructField("nearest_neighbours", T.StringType())])))

        nn_computed_df = df.withColumn("tup", F.struct(F.col('ID'), F.col('headline_embedding'), F.col('short_description_embedding'))) \
            .select('category', 'tup') \
            .groupBy('category') \
            .agg(F.collect_list(F.col('tup')).alias('group_data')) \
            .withColumn('knn_per_category', udf_knn_loops(F.col('group_data'))) \
            .select('category', F.explode(F.col('knn_per_category')))\
            .select(F.col('col.ID').alias('ID'),
                    F.col('col.scores').alias('scores'),
                    F.col('col.nearest_neighbours').alias('nearest_neighbours'))

        return nn_computed_df

    def distance_computation_v3(self, sqc, df):
        """
        1. concatenates the vectors for the 2 text fields, the one-hot encoded category field and min-max normalized popularity score
        2. normalizes the vectors to unit norm
        3. performs a O(n^2) comparisons after collecting all the vectors into memory. it is not scalable, but if we use a spark cluster there is
        an alternative using the crossJoin() approach as shown in distance_computation_v3_parallel()
        :param sqc: spark context
        :param df: dataframe
        :return: dataframe of [ID, 2 nearest scores, 2 nearest neighbours]
        """
        def feature_vec(category, popularity, headline, description):
            """
            generates a feature vector from all the fields
            :param category:
            :param popularity:
            :param headline:
            :param description:
            :return:
            """
            vec = np.concatenate([category, [popularity], headline, description])
            vec = vec / np.sqrt(np.sum(vec ** 2))
            return vec.tolist()

        udf_feature_vec = F.udf(feature_vec, T.ArrayType(T.FloatType()))

        id_feat_df = df.withColumn('feature_vec', udf_feature_vec('category_one_hot', 'normalized_popularity_score', 'headline_embedding',
                                                                  'short_description_embedding')).select('ID', 'feature_vec')

        data_mat = id_feat_df.collect()

        def knn_loops(data):
            """
            implements an O(n^2) comparison between elements in an array.
            :param data: list of type Row().
            :return: list of items of the form (ID, "score1,score1", "if of neighbour1,id of neighbour2")
            """
            print(len(data))
            score_list = list()
            for src in data:
                top2 = [1000, 1000]
                id2 = ["-1", "-1"]
                for dst in data:
                    if src.ID == dst.ID:
                        continue
                    dist = np.linalg.norm(np.array(src.feature_vec) - np.array(dst.feature_vec))
                    if dist < top2[0]:
                        top2[0] = dist
                        id2[0] = dst.ID
                    elif dist < top2[1]:
                        top2[1] = dist
                        id2[1] = dst.ID

                for idx, i in enumerate(top2):
                    top2[idx] = 0 if top2[idx] == 1000 else top2[idx]
                for idx, i in enumerate(id2):
                    id2[idx] = "" if id2[idx] == "-1" else id2[idx]
                score_list.append((src.ID, "{},{}".format(top2[0], top2[1]), "{},{}".format(id2[0], id2[1])))
            return score_list
        score_list = knn_loops(data_mat)
        score_df = sqc.createDataFrame(score_list, ['ID', 'scores', 'nearest_neighbours'])
        return score_df

    def distance_computation_v3_parallel(self, df, df_other):
        """
        """
        def dist(feature_vec_a, feature_vec_b):
            dist = np.linalg.norm(np.array(feature_vec_a) - np.array(feature_vec_b))
            # all three get the same weightage
            return float(dist)

        def knn(neighbour_list):
            """
            the group-by on the ID field gives the distances between that row and all rows. From among the distances top-2 are selected
            :param neighbour_list: list of distances for this row
            :return: a tuple of the 2 distances and 2 IDs corresponding to those distances. They are comma separated strings
            """
            NUM_NEAREST = 2
            nearest2 = sorted(neighbour_list, key=lambda a: a[0], reverse=False)[:NUM_NEAREST]
            dist_str = ",".join(list(map(lambda x: str(x[0]), nearest2)))
            id_str = ",".join(list(map(lambda x: str(x[1]), nearest2)))
            return dist_str, id_str

        # udf for computing distances
        udf_dist = F.udf(dist, T.FloatType())
        # udf to get the top 2 closest distances
        udf_knn = F.udf(knn, T.StructType([T.StructField("dist_str",T.StringType()), T.StructField("id_str", T.StringType())]))

        # the data frame processing step involving the join
        df_distance_calculated = df.repartition(10).crossJoin(df_other.repartition(10)) \
            .where(F.col('ID') != F.col('ID_other')) \
            .withColumn("dist", udf_dist('feature_vec', 'feature_other')) \
            .withColumn("tup", F.struct(F.col('dist'), F.col('ID_other'))) \
            .groupBy("ID") \
            .agg(F.collect_list('tup').alias('dist_against_others')) \
            .withColumn('2neighbour', udf_knn(F.col('dist_against_others'))) \
            .select('ID', F.col('2neighbour.dist_str').alias('scores'), F.col('2neighbour.id_str').alias('nearest_neighbour'))
        return df_distance_calculated
