# KATRec: Knowledge Aware aTtentive Sequential Recommendations
This is our TensorFlow implementation for the paper: Mehrnaz Amjadi, Danial Mohseni Taheri, Theja Tulabandhula (2021): KATRec: Knowledge Aware aTtentive Sequential Recommendations(https://arxiv.org/abs/2012.03323). 
Please cite our paper if you use the code or datasets.
The code is tested under a Linux desktop with tensorflow 1.15 and Python3.

# Datasets
The graph part of the preprocessed datasets for Amazon-book and Last-fm can be found from https://github.com/xiangwang1223/knowledge_graph_attention_network. We preprocessed the dataset. The *Data/Datasetname/kg_final* file is in the format of triplet (head/relation/tail). The sequential datasets that includes the time series interaction of users and items should be downloaded from the origin and paste in the *Model/data* folder. 

The sequential datasets that includes the time series interaction of users and items are downloaded from references below and preprocessed. Each line contains an *user id* and *item id* (starting from 1) meaning an interaction (sorted by timestamp).

Below, you can find the references for sequential datasets.
- Amazon-book: He, R. and McAuley, J., 2016, April. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web (pp. 507-517).
- Last-fm: Schedl, M., 2016, June. The lfm-1b dataset for music retrieval and recommendation. In Proceedings of the 2016 ACM on international conference on multimedia retrieval (pp. 103-110).

# Model Training
To train our model on amazon dataset ("in the Model folder"):
```
python run_amazon-book.sh
```
To train our model on last-fm dataset ("in the Model folder"):
```
python run_last-fm.sh
```
