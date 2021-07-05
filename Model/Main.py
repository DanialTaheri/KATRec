# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from utility.helper import *
#from batch_test import *
from time import time
from utility.loader_kgat import KGAT_loader
import modeling_Bert
import optimization
import tensorflow as tf
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters for KGAT
flags.DEFINE_string(
    "model_type", None,
    "Specify a loss type.")

flags.DEFINE_integer(
    "pretrain", 0,
    "0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.")

flags.DEFINE_float(
    "lr", 0.0001,
    "learning rate for TransR.")

flags.DEFINE_integer(
    "batch_size", 1024,
    "batch size for colab KG.")

flags.DEFINE_integer(
    "kge_size", 64,
    "embedding size in TransR.")

flags.DEFINE_integer(
    "n_instances", 30,
    "number of batches")

flags.DEFINE_integer(
    "batch_size_kg", 2048,
    "batch size for TransR.")

flags.DEFINE_string(
    "layer_size", None,
    "Number of neurons in each layer TransR.")

flags.DEFINE_string(
    "alg_type", None,
    "specifies the type of graph convolutional layer from {bi}.")

flags.DEFINE_string(
    "regs", None,
    "reg coeficient for TransR and colab KG.")

flags.DEFINE_integer(
    "embed_size", 64,
    "embed size for Colab KG.")


flags.DEFINE_string(
    "node_dropout", None,
    "nod dropout for each deep layer")

flags.DEFINE_string(
    "mess_dropout", None,
    "message dropout for each deep layer")

flags.DEFINE_string(
    "adj_uni_type", 'sum',
    "Specify a loss type (uni, sum).")

flags.DEFINE_string(
    "adj_type", 'si',
    "Specify the type of the adjacency (laplacian) matrix from {bi, si}.")


flags.DEFINE_string(
    "dataset_name", None,
    "Name of dataset for KG.")


flags.DEFINE_string(
    "Ks", None,
    "Output sizes of every layer.")

flags.DEFINE_string(
    "data_path_kg", ".../Data/",
    "Input data path")


flags.DEFINE_string(
    "weights_path", None,
    "Store model path.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("signature", 'default', "signature_name")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_bool("do_train", False, 
                  "Whether to run training.")

flags.DEFINE_bool("do_eval", False, 
                  "Whether to run eval on the dev set.")

flags.DEFINE_integer("num_train_steps", 100000, 
                     "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000,
                     "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, 
                     "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, 
                  "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_pop_random", False, 
                  "use pop random negative samples")

flags.DEFINE_bool("use_KG_connection", True, 
                  "use KG embeddings for initiating BERT")

flags.DEFINE_bool("use_KG_attention", True, 
                  "use attention for KG part")

flags.DEFINE_bool("use_token_entity_concat", True, 
                  "use concatenation of token and entity in the output layer")

flags.DEFINE_string("vocab_filename", None, 
                    "vocab filename")

flags.DEFINE_string("user_history_filename", None, 
                    "user history filename")

flags.DEFINE_bool("plot_attention", True,
                  "plot the attention weights")


class EvalHooks(tf.train.SessionRunHook):
    def __init__(self):
        tf.logging.info('run init')

    def begin(self):
        self.valid_user = 0.0
        self.valid_user_3rdq, self.valid_user_2ndq, self.valid_user_1stq = 0.0, 0.0, 0.0
        self.ndcg_1_3rdq, self.ndcg_1_2ndq, self.ndcg_1_1stq = 0.0, 0.0, 0.0
        self.hit_1_3rdq, self.hit_1_2ndq, self.hit_1_1stq = 0.0, 0.0, 0.0
        self.ndcg_5_3rdq, self.ndcg_5_2ndq, self.ndcg_5_1stq = 0.0, 0.0, 0.0
        self.hit_5_3rdq, self.hit_5_2ndq, self.hit_5_1stq = 0.0, 0.0, 0.0
        self.ndcg_10_3rdq, self.ndcg_10_2ndq, self.ndcg_10_1stq = 0.0, 0.0, 0.0
        self.hit_10_3rdq, self.hit_10_2ndq, self.hit_10_1stq = 0.0, 0.0, 0.0
        self.ap_3rdq, self.ap_2ndq, self.ap_1stq = 0.0, 0.0, 0.0
        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ap = 0.0
        
        self.all_attn_map = []
        self.all_input_ids = []

        np.random.seed(12345)

        self.vocab = None

        if FLAGS.user_history_filename is not None:
            print('load user history from :' + FLAGS.user_history_filename)
            with open(FLAGS.user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if FLAGS.vocab_filename is not None:
            print('load vocab from :' + FLAGS.vocab_filename)
            with open(FLAGS.vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]

    def end(self, session):
#         print(
#             "ndcg@1:{}, hit@1:{}， ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{},\
#             ndcg@1_3q:{}, hit@1_3q:{}， ndcg@5_3q:{}, hit@5_3q:{}, ndcg@10_3q:{}, hit@10_3q:{}, ap_3q:{}, valid_user_3q:{},\
#             ndcg@1_2q:{}, hit@1_2q:{}， ndcg@5_2q:{}, hit@5_2q:{}, ndcg@10_2q:{}, hit@10_2q:{}, ap_2q:{}, valid_user_2q:{},\
#             ndcg@1_1q:{}, hit@1_1q:{}， ndcg@5_1q:{}, hit@5_1q:{}, ndcg@10_1q:{}, hit@10_1q:{}, ap_1q:{}, valid_user_1q:{}".
#             format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
#                    self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
#                    self.ndcg_10 / self.valid_user, self.hit_10 / self.valid_user,
#                    self.ap / self.valid_user,self.valid_user,
#                    self.ndcg_1_3rdq / self.valid_user_3rdq, self.hit_1_3rdq / self.valid_user_3rdq,
#                    self.ndcg_5_3rdq / self.valid_user_3rdq, self.hit_5_3rdq / self.valid_user_3rdq,
#                    self.ndcg_10_3rdq / self.valid_user_3rdq, self.hit_10_3rdq / self.valid_user_3rdq,
#                    self.ap_3rdq / self.valid_user_3rdq, self.valid_user_3rdq,
#                    self.ndcg_1_2ndq / self.valid_user_2ndq, self.hit_1_2ndq / self.valid_user_2ndq,
#                    self.ndcg_5_2ndq / self.valid_user_2ndq, self.hit_5_2ndq / self.valid_user_2ndq,
#                    self.ndcg_10_2ndq / self.valid_user_2ndq, self.hit_10_2ndq / self.valid_user_2ndq,
#                    self.ap_2ndq / self.valid_user_2ndq, self.valid_user_2ndq,
#                    self.ndcg_1_1stq / self.valid_user_1stq, self.hit_1_1stq / self.valid_user_1stq,
#                    self.ndcg_5_1stq / self.valid_user_1stq, self.hit_5_1stq / self.valid_user_1stq,
#                    self.ndcg_10_1stq / self.valid_user_1stq, self.hit_10_1stq / self.valid_user_1stq,
#                    self.ap_1stq / self.valid_user_1stq, self.valid_user_1stq))
        output = (self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
        self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
        self.ndcg_10 / self.valid_user,
        self.hit_10 / self.valid_user, self.ap / self.valid_user)
        with open(os.path.join(FLAGS.checkpointDir, 'log.txt'), 'a') as f:
            f.write(str(output) + '\n')
        dic = {}
        feature_dicts_with_att = []
        for nb_batch in range(len(self.all_input_ids)):
            for elem, elem_att in zip(self.all_input_ids[nb_batch], self.all_attn_map[nb_batch]):
                dic["attn"] = elem_att.astype("float16")
                dic["ids"] = elem
                feature_dicts_with_att.append(dic)
        path=FLAGS.checkpointDir+'/attn.pkl'
        print("writing attention maps to: {} ".format(path))
        with tf.gfile.GFile(path, 'wb') as f:
            pickle.dump(feature_dicts_with_att, f, -1)
 

    def before_run(self, run_context):
        #tf.logging.info('run before run')
        #print('run before_run')
        variables = tf.get_collection('eval_sp')
        return tf.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        #tf.logging.info('run after run')
        #print('run after run')
        masked_lm_log_probs, input_ids, masked_lm_ids, info, attn_map = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, FLAGS.max_predictions_per_seq, masked_lm_log_probs.shape[1]))
        
        if FLAGS.plot_attention:
            self.all_attn_map.append(attn_map)
            self.all_input_ids.append(input_ids)
        
        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            rated.add(masked_lm_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            item_idx = [masked_lm_ids[idx][0]]
            # here we need more consideration
            masked_lm_log_probs_elem = masked_lm_log_probs[idx, 0]  
            size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
            
            if FLAGS.use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 101:
                        sampled_ids = np.random.choice(self.ids, 101, replace=False, p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:101]
            else:
                # print("evaluation random -> ")
                for _ in range(100):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions = -masked_lm_log_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[0]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            if rank < 1:
                self.ndcg_1 += 1
                self.hit_1 += 1
            if rank < 5:
                self.ndcg_5 += 1 / np.log2(rank + 2)
                self.hit_5 += 1
            if rank < 10:
                self.ndcg_10 += 1 / np.log2(rank + 2)
                self.hit_10 += 1

            self.ap += 1.0 / (rank + 1)
            
            user_length = len(self.user_history["user_" + str(info[idx][0])][0])
            user_length_threshold = [73, 143, 403]
            #75% quantile
            if user_length < user_length_threshold[2]:
                self.valid_user_3rdq += 1

                if rank < 1:
                    self.ndcg_1_3rdq += 1
                    self.hit_1_3rdq += 1
                if rank < 5:
                    self.ndcg_5_3rdq += 1 / np.log2(rank + 2)
                    self.hit_5_3rdq += 1
                if rank < 10:
                    self.ndcg_10_3rdq += 1 / np.log2(rank + 2)
                    self.hit_10_3rdq += 1

                self.ap_3rdq += 1.0 / (rank + 1)
            
            #50% quantile
            if user_length < user_length_threshold[1]:
                self.valid_user_2ndq += 1

                if rank < 1:
                    self.ndcg_1_2ndq += 1
                    self.hit_1_2ndq += 1
                if rank < 5:
                    self.ndcg_5_2ndq += 1 / np.log2(rank + 2)
                    self.hit_5_2ndq += 1
                if rank < 10:
                    self.ndcg_10_2ndq += 1 / np.log2(rank + 2)
                    self.hit_10_2ndq += 1

                self.ap_2ndq += 1.0 / (rank + 1)
      
            #25% quantile
            if user_length < user_length_threshold[0]:
                self.valid_user_1stq += 1

                if rank < 1:
                    self.ndcg_1_1stq += 1
                    self.hit_1_1stq += 1
                if rank < 5:
                    self.ndcg_5_1stq += 1 / np.log2(rank + 2)
                    self.hit_5_1stq += 1
                if rank < 10:
                    self.ndcg_10_1stq += 1 / np.log2(rank + 2)
                    self.hit_10_1stq += 1
                self.ap_1stq += 1.0 / (rank + 1)

def model_fn_builder(bert_config, kgat_config, pretrain_data, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size, FLAGS):
    
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name,
                                                         features[name].shape))
        
        h = features["h"]
        r = features["r"]
        pos_t = features["pos_t"]
        neg_t = features["neg_t"]
        h = tf.reshape(h, [-1])
        r = tf.reshape(r, [-1])
        pos_t = tf.reshape(pos_t, [-1])
        neg_t = tf.reshape(neg_t, [-1])
        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = modeling_Bert.KATRecModel(
            head=h, 
            relation=r, 
            pos_t=pos_t, 
            neg_t=neg_t,
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            user_ids=info,
            data_config=kgat_config,
            pretrain_data=pretrain_data,
            FLAGS=FLAGS,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)

        attn_map = model.get_all_attention_weights()
        
        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
             bert_config,
             model.get_sequence_output(),
             model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
             masked_lm_weights, model.get_kg_embedding_table(), model.get_kg_user_embedding(), info)
        
        kgat_loss = model.get_loss_TransR()
        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling_Bert.get_assignment_map_from_checkpoint(
                 tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("opt") as scope:
                train_op1 = optimization.create_optimizer(kgat_loss, learning_rate,
                                                         num_train_steps,
                                                         num_warmup_steps, use_tpu)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):     
                train_op2 = optimization.create_optimizer(total_loss, learning_rate,
                                                         num_train_steps,
                                                         num_warmup_steps, use_tpu)

            train_ops = tf.group(train_op2, train_op1)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=kgat_loss+total_loss,
                train_op=train_ops,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(kgat_loss, masked_lm_example_loss, masked_lm_log_probs,
                          masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss,
                                                    [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)
                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            tf.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.add_to_collection('eval_sp', input_ids)
            tf.add_to_collection('eval_sp', masked_lm_ids)
            tf.add_to_collection('eval_sp', info)
            tf.add_to_collection('eval_sp', attn_map)
            
            eval_metrics = metric_fn(kgat_loss, masked_lm_example_loss,
                                     masked_lm_log_probs, masked_lm_ids,
                                     masked_lm_weights)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" %
                             (mode))

        return output_spec

    return model_fn

def get_masked_lm_output(bert_config, input_tensor, bert_output_weights, positions,
                         label_ids, label_weights, kg_output_weights, kg_input_tensor, user_pos):
    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    
    input_tensor = gather_indexes(input_tensor, positions)
    #user_pos = tf.repeat(user_pos, repeats=[positions.get_shape().as_list()[1]], axis=1)
    #kg_input_tensor = gather_indexes(kg_input_tensor, user_pos)
    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            #input_tensor = tf.concat([input_tensor, kg_input_tensor], axis=1)
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling_Bert.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling_Bert.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling_Bert.layer_norm(input_tensor)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # Concating kg and bert outputs
        #Only based on cancatenation
        ##############################
        # Information fusion is concatenation
        if FLAGS.use_token_entity_concat:
            output_weights = tf.concat([bert_output_weights, kg_output_weights], 1)
            output_weights = tf.layers.dense(
                    output_weights,
                    units=bert_config.hidden_size,
                    activation=modeling_Bert.get_activation(bert_config.hidden_act),
                    use_bias=True,
                    kernel_initializer=modeling_Bert.create_initializer(
                        bert_config.initializer_range))
        else:
            output_weights = bert_output_weights
        
        output_bias = tf.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)


        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)

def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling_Bert.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor





os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_pretrained_data(FLAGS):
    pre_model = 'mf'
    pretrain_path = './pretrain/%s/%s.npz' % (FLAGS.dataset_name, pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data

def input_fn_builder(input_files,
                     all_v_list,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "h":
            tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "r":
            tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "pos_t":
            tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "neg_t":
            tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "info":
            tf.FixedLenFeature([1], tf.int64),  #[user]
            "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

            # `cycle_length` is the number of parallel files that get read.
            #cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            #d = d.apply(
            #    tf.contrib.data.parallel_interleave(
            #        tf.data.TFRecordDataset,
            #        sloppy=is_training,
            #        cycle_length=cycle_length))
            #d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)


        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size) #Combines consecutive elements of this dataset into batches.
        return d

    return input_fn

def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example



        


if __name__ == '__main__':
    
    tf.reset_default_graph()
    # get argument settings.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(2019)
    np.random.seed(2019)
    #args = parse_args()


    print(FLAGS.dataset_name)
    """
    *********************************************************
    Load Data from data_generator function.
    """
    if FLAGS.model_type in ['kg']:
        data_generator = KGAT_loader(FLAGS=FLAGS, path=FLAGS.data_path_kg + FLAGS.dataset_name)
    batch_test_flag = False
    
    # data_generator is in the batch_test
    Kgatconfig = dict()
    Kgatconfig['n_users'] = data_generator.n_users
    Kgatconfig['n_items'] = data_generator.n_items
    Kgatconfig['n_relations'] = data_generator.n_relations
    Kgatconfig['n_entities'] = data_generator.n_entities
    
    if FLAGS.model_type in ['kg']:
        "Load the laplacian matrix."
        Kgatconfig['A_in'] = sum(data_generator.lap_list)

        "Load the KG triplets."
        Kgatconfig['all_h_list'] = data_generator.all_h_list
        Kgatconfig['all_r_list'] = data_generator.all_r_list
        Kgatconfig['all_t_list'] = data_generator.all_t_list
        Kgatconfig['all_v_list'] = data_generator.all_v_list

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if FLAGS.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(FLAGS)
    else:
        pretrain_data = None
        
    """
    **********************************************************
    Bert
    
    """

    FLAGS.checkpointDir = FLAGS.checkpointDir + FLAGS.signature
    print('checkpointDir:', FLAGS.checkpointDir)

    if not FLAGS.do_train and not FLAGS.do_eval:
         raise ValueError(
             "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling_Bert.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
         train_input_files.extend(tf.gfile.Glob(input_pattern))

    test_input_files = []
    if FLAGS.test_input_file is None:
         test_input_files = train_input_files
    else:
        for input_pattern in FLAGS.test_input_file.split(","):
             test_input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** train Input Files ***")
    for input_file in train_input_files:
         tf.logging.info("  %s" % input_file)

    tf.logging.info("*** test Input Files ***")
    for input_file in train_input_files:
         tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None

     #is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.RunConfig(
         model_dir=FLAGS.checkpointDir,
         save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    
    if FLAGS.vocab_filename is not None:
        with open(FLAGS.vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    
    model_fn = model_fn_builder(
        bert_config=bert_config,
        kgat_config=Kgatconfig,
        pretrain_data=pretrain_data,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.lr,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        item_size=item_size,
        FLAGS=FLAGS)
    



    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    #######################
    #    New Part
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": FLAGS.batch_size
        })
    print(FLAGS.do_train)
    

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.batch_size)

    train_input_fn = input_fn_builder(
        input_files=train_input_files,
        all_v_list=data_generator.all_v_list,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)


    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.batch_size)

    eval_input_fn = input_fn_builder(
        input_files=test_input_files,
        all_v_list=data_generator.all_v_list,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)


        #tf.logging.info('special eval ops:', special_eval_ops)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None,
        hooks=[EvalHooks()])
    
    
    tf.estimator.train_and_evaluate(estimator, train_spec,
                                   eval_spec)
    

    
    
    
