from __future__ import print_function
import sys
import copy
import random
import numpy as np
from collections import defaultdict
import os


def data_partition(fname, FLAGS):
    print("inside utility function")
    usernum = 1
    itemnum = 1
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    path = FLAGS.data_path_kgat + FLAGS.dataset_name
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
                #print " ".join(map(str, user_train[user]))
 
            
        with open(path + "/"+ "train.txt", "a") as train_file:
            train_file.write(str(user) + " " +" ".join([str(x) for x in user_train[user]]) + "\n")

            
        with open(path + "/"+ "test.txt", "a") as test_file:
            test_file.write(str(user) + " " +" ".join([str(x) for x in user_test[user]]) + "\n")

            
    return [user_train, user_valid, user_test, usernum, itemnum]

