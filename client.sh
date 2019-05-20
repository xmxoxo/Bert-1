#!/usr/bin/env bash
export BERT_BASE_DIR=/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12
export GLUE_DIR=/home/new/Bert/data
export MODEL_DIR=/home/new/Bert/output

python client.py \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/model.ckpt-6072 \
  --max_seq_length=128 \
  --predict_batch_size=2 \
  --output_dir= MODEL_DIR

