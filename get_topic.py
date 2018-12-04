# coding: utf-8
from __future__ import print_function
import json
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'classification/data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'classification/checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

data_path = "classification/data/rec_news/user_click_data.txt"

result_path = "classification/data/rec_news/news_topic.json"


def get_news_users_set():
    news = {}
    user_news = {}
    with open(data_path) as f:
        for line in f.readlines():
            line = line.split("\t")
            user_id = line[0]
            news_id = line[1]
            read_time = line[2]
            title = line[3]
            content = line[4]
            news_time = line[5]
            if title != '404' or content != 'null':
                user_item = {
                    'news_id': news_id,
                    'read_time': read_time
                }
                if user_id not in user_news:
                    user_news[user_id] = [user_item]
                else:
                    user_news[user_id].append(user_item)
                news_item = {
                    'title': title,
                    'content': content,
                    'news_time': news_time
                }
                if news_id not in news:
                    news[news_id] = news_item
                    news[news_id]['count'] = 1
                else:
                    news[news_id]['count'] += 1
    return user_news, news


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    _, news = get_news_users_set()
    count = 0
    for k, v in news.items():
        topic = cnn_model.predict(v["content"])
        if count % 100 == 0:
            end = "\n"
        else:
            end = "\t"
        count += 1
        print(k, topic, end=end)
        news[k]["topic"] = topic
    # save topic to json
    json_obj = json.dumps(str(news))
    for k,v in news.items():
        print(k,v)
        break
    with open(result_path, 'w', decoding="utf-8") as f:
        f.write(json_obj)
    print("completed!")
