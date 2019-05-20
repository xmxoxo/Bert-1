from __future__ import print_function


import requests
from classifier import *
import time
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

endpoint = 'http://127.0.0.1:8500'


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

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

    d = tf.data.TFRecordDataset(input_file)
    # print('arrive input_fn 3')
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
        # print('arrive input_fn 4')

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=FLAGS.predict_batch_size,
            drop_remainder=drop_remainder))
    # print('finish input_fn ')

    return d



class Client:
    def __init__(self):
        self.processor = MyProcessor()

    def sort_and_retrive(self, predictions, qa_pairs):
        res = []
        for prediction, qa in zip(predictions, qa_pairs):
            res.append((prediction[1], qa))
        res.sort(reverse=True)
        return res

    def preprocess(self, sentences):
        save_path = os.path.join(FLAGS.data_dir, "pred.tsv")
        with open(save_path, 'w') as fout:
            out_line = '0' + '\t' + ' ' + '\n'
            fout.write(out_line)
            for sentence in sentences:
                out_line = '0'+'\t' + sentence + '\n'
                fout.write(out_line)

    def predict(self, sentences):

        self.preprocess(sentences)

        predict_examples = self.processor.get_pred_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        label_list = self.processor.get_labels()
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer, predict_file)
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_dataset = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        iterator = predict_dataset.make_one_shot_iterator()

        next_element = iterator.get_next()

        inputs = ["label_ids", "input_ids", "input_mask", "segment_ids"]
        for input in inputs:
            next_element[input] = next_element[input].numpy().tolist()

        json_data = {"model_name": "default", "data": next_element}

        start = time.time()
        result = requests.post(endpoint, json=json_data)
        cost = time.time() - start
        print('total time cost: %s s' % cost)

        result = dict(result.json())

        output = [np.argmax(i)-1 for i in result['output']]
        return output

if __name__ == '__main__':
    client = Client()
    msg = ["电池一直用可以用半天，屏幕很好。","机是正品，用着很流畅，618活动时买的，便宜了不少！",""]
    prediction = client.predict(msg)
    # print('probability: %s'%prediction)
    print(prediction)

