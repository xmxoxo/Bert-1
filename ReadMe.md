## 简单高效的Bert中文文本分类模型开发和部署

### 准备环境工作

- **操作系统**：Linux

- **TensorFlow Version**：1.13.1，动态图模式
- **GPU**：我的服务器是Tesla P4 8G GPU，文档后面有显存不足的解决方案
- **TensorFlow Serving**：[simple-tensorflow-serving](<https://stfs.readthedocs.io/en/latest/quick_start.html>)
- **依赖库**：requirements.txt

### 目录结构说明

![Evj50e.png](https://s2.ax1x.com/2019/05/20/Evj50e.png)

- bert是官方[源码](https://github.com/google-research/bert)
- data是数据，来自[项目](https://github.com/xmxoxo/BERT-train2deploy)，文本的3分类问题
- train.sh、classifier.py 训练文件
- export.sh、export.py导出TF serving的模型
- client.sh、client.py、file_base_client.py 处理输入数据并向部署的TF serving的模型发出请求，打印输出结果

### 训练代码

和[项目](https://github.com/xmxoxo/BERT-train2deploy)基本一致，特殊的地方我会指出。

1. 写一个自己的文本处理器。有两点需要**注意**：1，改写label 2，把create_examples改成了共有方法，因为我们后面要调用。3，file_base的时候注意跳过第一行，文件数据的第一行是title

   ```python
   class MyProcessor(DataProcessor):
   
       def get_test_examples(self, data_dir):
           return self.create_examples(
               self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
   
       def get_train_examples(self, data_dir):
           """See base class."""
           return self.create_examples(
               self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
   
       def get_dev_examples(self, data_dir):
           """See base class."""
           return self.create_examples(
               self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
   
       def get_pred_examples(self, data_dir):
           return self.create_examples(
               self._read_tsv(os.path.join(data_dir, "pred.tsv")), "pred")
   
       def get_labels(self):
           """See base class."""
           return ["-1", "0", "1"]
   
       def create_examples(self, lines, set_type, file_base=True):
           """Creates examples for the training and dev sets. each line is label+\t+text_a+\t+text_b """
           examples = []
           for (i, line) in tqdm(enumerate(lines)):
   
               if file_base:
                   if i == 0:
                       continue
   
               guid = "%s-%s" % (set_type, i)
               text = tokenization.convert_to_unicode(line[1])
               if set_type == "test" or set_type == "pred":
                   label = "0"
               else:
                   label = tokenization.convert_to_unicode(line[0])
               examples.append(
                   InputExample(guid=guid, text_a=text, label=label))
           return examples
   
   ```

2. 其他的训练代码，照抄官方的就行

3. 可以直接运行train.sh，**注意修改对应的路径**

4. 生成的ckpt文件在output路径下

### 导出模型

主要代码如下，生成的pb文件在api文件夹下

```python
def serving_input_receiver_fn():
    input_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='segment_ids')
    label_ids = tf.placeholder(dtype=tf.int64, shape=[None, ], name='unique_ids')

    receive_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids,
                       'label_ids': label_ids}
    features = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, "label_ids": label_ids}
    return tf.estimator.export.ServingInputReceiver(features, receive_tensors)

estimator.export_savedmodel(FLAGS.serving_model_save_path, serving_input_receiver_fn)
```

### TensorFlow Serving部署

一键部署：

```bash
simple_tensorflow_serving --model_base_path="./api"
```

正常启动终端界面：

![EvO7HH.png](https://s2.ax1x.com/2019/05/20/EvO7HH.png)

浏览器访问界面：

![EvOouD.png](https://s2.ax1x.com/2019/05/20/EvOouD.png)

这部分认真阅读simple-tensorflow-serving的[文档](<https://stfs.readthedocs.io/en/latest/quick_start.html>)

### 本地请求代码

分为两种，一种是读取文件的，就是要预测的文本是tsv文件的，叫做file_base_client.py，另一个直接输入文本的是client.py。首先更改input_fn_builder，返回dataset，然后从dataset中取数据，转换为list格式，传入模型，返回结果。

正常情况下的运行结果：

![Exkyz4.png](https://s2.ax1x.com/2019/05/20/Exkyz4.png)

### 问题解答

- 训练的显存不足怎么办

  答：按照官方的建议，调小max_seq_length和train_batch_size

### TODO LIST

- [ ] 接入Docker
- [ ] 微信端交互代码



