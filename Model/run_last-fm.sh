model_type="kg"
pretrain=-1
lr=0.0001
batch_size=128
kge_size=32
batch_size_kg=1024
layer_size='[64]'
alg_type='bi'
regs='[1e-4,1e-4]'
embed_size=64
node_dropout=[0.1]
mess_dropout='[0.2,0.2,0.2]'
adj_uni_type='sum'
adj_type='si'
Ks='[20, 40, 60, 80, 100]'
test_flag="part"
weights_path=''
data_path_kg="../Data/"
proj_path=''

CKPT_DIR="BERTresults"
dataset_name="last-fm"
max_seq_length=200
masked_lm_prob=0.2
max_predictions_per_seq=20

dim=128
num_train_steps=400000

prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=10
pool_size=10

concatenation=True
KG_connection=True
KG_attention=True


signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}-kges${embed_size}"

Experiments="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-mpps${max_predictions_per_seq}-msl${max_seq_length}-kges${embed_size}-pretrain${pretrain}-KGlayer${layer_size}-KGatt${KG_attention}-KGconn${KG_connection}-concat${concatenation}"

python -u gen_data_fin.py \
    --dataset_name=${dataset_name} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --mask_prob=${mask_prob} \
    --dupe_factor=${dupe_factor} \
    --masked_lm_prob=${masked_lm_prob} \
    --prop_sliding_window=${prop_sliding_window} \
    --signature=${signature} \
    --pool_size=${pool_size} \
    --model_type=${model_type}\
    --adj_type=${adj_type}\
    --batch_size=${batch_size}


CUDA_VISIBLE_DEVICES=0 python -u Main.py \
    --model_type=${model_type} \
    --use_KG_attention=${KG_attention} \
    --use_KG_connection=${KG_connection}\
    --use_token_entity_concat=${concatenation}\
    --dataset_name=${dataset_name}\
    --data_path_kg=${data_path_kg} \
    --pretrain=${pretrain} \
    --lr=${lr} \
    --batch_size=${batch_size} \
    --kge_size=${kge_size} \
    --batch_size_kg=${batch_size_kg} \
    --layer_size=${layer_size} \
    --alg_type=${alg_type} \
    --regs=${regs}\
    --embed_size=${embed_size}\
    --node_dropout=${node_dropout}\
    --mess_dropout=${mess_dropout}\
    --adj_uni_type=${adj_uni_type}\
    --adj_type=${adj_type}\
    --Ks=${Ks}\
    --weights_path=${weights_path}\
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
