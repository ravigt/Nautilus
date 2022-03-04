## Approach
- The solution is implemented as a knn match where n=2
- to compare distances the `headline` `category` and `short description` columns of the csv file were chosen
- distance is measured separately for the three columns. For `headline` and `short description` columns L2 norm was used as the distance . For
 `category` column absolute match was used as the distance measure.
 - the average of the L2 norm for the 2 text columns is the final distance
 - we can give different weightage to the 2 distances but it has not been done. However `category` was given the highest weightage. This means, if
  the `category` does not match the distance is infinite.
 
 ## Feature generation
 - distilBERT was used as the pretrained language model
 - each text segment was separately converted into 798 dimensional embedding. this means for a given row, the `headline` and `short descriptions
 ` have their own embedding
 ### Alternatives
 - a simpler approach is break to the text into a bag of words after skipping all the stop words. then use jaccard distance as the distance
  measure. This is memory efficient, proportional to the number of words. But this fails to capture semantic similarity.
 - converting text to a binary encoded vector is also not efficient as there can be huge number of dimensions. Jaccard distance between
 text blocks would give similar results with lower memory and compute footprint. Vectors are useful if we want to create weights for each words
 . But if weights are not there, then it is same as jaccard distance but with more memory. More over vectores are sparse.
 -  Preference was given to better accuracy in terms of semantic similarity and hence the embedding approach was used although there is additional
  compute cost due to the use of a deep learning model.

 ## processing at sale
 - pyspark dataframe processing with inbuilt column operators was preferred where ever possible
 - custom logic was implemented as UDFs.
 - nxn distance computation was implemented in ```distance_computation()``` function. This function leverages cross join to perform nxn distance
  computation and then to take the top 2 smallest distances. for this to work we need a proper spark cluster with separate executor nodes.
  - as `category` field involves an exact match, instead of using the match in a distance formula it was used as a groupBy key and this gives the
   same result while leveraging the spark operators
 - another distance computation function called ```distance_computation_v2()``` was also implemented. This function performs a groupBy on the
  `category` and within each category group, implements a nxn distance computation keeping only the top 2 values. This means the parallelism is
   achieved at the `category` level. This is inefficient as `category` based groups are imbalanced. ```distance_computation()``` is better but
    would need a spark cluster. shuffles are inevitable
 
 ## writing the output
 - owing to the heap size requirements, the dataframe created after the knn computations was written to a directory as .parquet files.
 - these parquet files were then stitched into the .csv file
 - as spark does not support special characters in the column headers, the header names are slightly different from the names specified in the
  required output format
 
 ## performance hacks
 - as a GPU was not available, the text size (MAX_TEXT) for the text fields was curtailed to 5 words. This is to reduce the distilBERT compute
  time as far as possibe on a laptop with a CPU and limited memory.
 - even for short text segments of a few words, we get a 798 dimension vector. this means data bloats and hence memory requirements increase.
 - unfortunately these requirements were not met on the compute machine and hence a smaller dataset of 10k entries was used for processing
  instead of the full dataset of ~200k entries
  
 ## running the code
 - there are only 3 files. the imports will show the requirements
 - in ```processing_pipeline.py``` set ```file_path``` to the input csv file over which the nearest neighbours will be computed
 - in ```prepare_output.py``` set ```file_path``` to the input csv used above and set ```file_path_nearest_neighbours``` to the path to which the
  parquet files were written
 - run ```processing_pipeline.py``` to compute the neighbours
 - run ```prepare_output.py``` to join the original data with the nearest neighbours data to produce the result csv file