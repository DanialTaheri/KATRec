# KATRec: Knowledge Aware aTtentive Sequential Recommendations
This is our TensorFlow implementation for the paper: Mehrnaz Amjadi, Danial Mohseni Taheri, Theja Tulabandhula (2021): KATRec: Knowledge Aware aTtentive Sequential Recommendations(https://arxiv.org/abs/2012.03323). 
Please cite our paper if you use the code or datasets.
The code is tested under a Linux desktop with tensorflow 1.15 and Python3.

# Datasets
The graph part of the preprocessed datasets are included in the *Data/kg_final* folder in the format of triplet (head/relation/tail), and sequential datasets are included in the *Model/data*, where each line contains an *user id* and *item id* (starting from 1) meaning an interaction (sorted by timestamp).

# Model Training
To train our model on amazon dataset ("in the Model folder"):
```
python run_amazon-book.sh

```
To train our model on last-fm dataset ("in the Model folder"):
```
python run_last-fm.sh

```
