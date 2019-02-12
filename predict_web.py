from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from keras.utils import plot_model

import pandas as pd
import numpy as np
import re
import tensorflow as tf

from flask import Flask, jsonify, render_template, request


class TranslateModel(object):
    def __init__(self):
        global graph
        self.graph = tf.get_default_graph()
        self.NUM_SAMPLES = 10000
        # 英语翻译为中文
        self.en2zh_encoder_input, self.en2zh_decoder_output, self.en2zh_input_dict, self.en2zh_target_dict, self.en2zh_target_dict_reverse, self.en2zh_INUPT_LENGTH = self.data_util(
            english_to_chinese=True)
        self.en2zh_model_train = load_model('model/en2zh_model.h5')
        self.en2zh_encoder_infer = load_model('model/en2zh_encoder.h5')
        self.en2zh_decoder_infer = load_model('model/en2zh_decoder.h5')

        # 中文翻译为英语
        self.zh2en_encoder_input, self.zh2en_decoder_output, self.zh2en_input_dict, self.zh2en_target_dict, self.zh2en_target_dict_reverse, self.zh2en_INUPT_LENGTH = self.data_util(
            english_to_chinese=False)
        self.zh2en_model_train = load_model('model/zh2en_model.h5')
        self.zh2en_encoder_infer = load_model('model/zh2en_encoder.h5')
        self.zh2en_decoder_infer = load_model('model/zh2en_decoder.h5')

    def data_util(self, english_to_chinese=True):
        data_path = 'data/translate2048.txt'
        df = pd.read_table(data_path, header=None).iloc[:self.NUM_SAMPLES, :, ]
        # 去掉标点符号
        df.replace('[,.!?，。！？]', '', regex=True, inplace=True)
        # df.replace('[.!?。！？]','', regex=True, inplace=True)
        # 全部转小写
        df[0] = df[0].apply(lambda x: x.lower())

        if english_to_chinese:
            df.columns = ['inputs', 'targets']
        else:
            df.columns = ['targets', 'inputs']

        df['targets'] = df['targets'].apply(lambda x: '\t' + x + '\n')

        input_texts = df.inputs.values.tolist()
        target_texts = df.targets.values.tolist()

        input_characters = sorted(list(set(df.inputs.unique().sum())))
        target_characters = sorted(list(set(df.targets.unique().sum())))

        INUPT_LENGTH = max([len(i) for i in input_texts])
        OUTPUT_LENGTH = max([len(i) for i in target_texts])
        INPUT_FEATURE_LENGTH = len(input_characters)
        OUTPUT_FEATURE_LENGTH = len(target_characters)

        encoder_input = np.zeros((self.NUM_SAMPLES, INUPT_LENGTH, INPUT_FEATURE_LENGTH))
        decoder_input = np.zeros((self.NUM_SAMPLES, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH))
        decoder_output = np.zeros((self.NUM_SAMPLES, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH))

        input_dict = {char: index for index, char in enumerate(input_characters)}
        input_dict_reverse = {index: char for index, char in enumerate(input_characters)}
        target_dict = {char: index for index, char in enumerate(target_characters)}
        target_dict_reverse = {index: char for index, char in enumerate(target_characters)}

        for seq_index, seq in enumerate(input_texts):
            for char_index, char in enumerate(seq):
                encoder_input[seq_index, char_index, input_dict[char]] = 1

        for seq_index, seq in enumerate(target_texts):
            for char_index, char in enumerate(seq):
                decoder_input[seq_index, char_index, target_dict[char]] = 1.0
                if char_index > 0:
                    decoder_output[seq_index, char_index - 1, target_dict[char]] = 1.0

        return encoder_input, decoder_output, input_dict, target_dict, target_dict_reverse, INUPT_LENGTH

    def predict(self, source, encoder_inference, decoder_inference, n_steps, features, target_dict,
                target_dict_reverse):
        with self.graph.as_default():
            # 先通过推理encoder获得预测输入序列的隐状态
            state = encoder_inference.predict(source)
            # 第一个字符'\t',为起始标志
            predict_seq = np.zeros((1, 1, features))
            predict_seq[0, 0, target_dict['\t']] = 1

            output = ''
            # 开始对encoder获得的隐状态进行推理
            # 每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
            for i in range(n_steps):  # n_steps为句子最大长度
                # 给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
                yhat, h, c = decoder_inference.predict([predict_seq] + state)
                # 注意，这里的yhat为Dense之后输出的结果，因此与h不同
                char_index = np.argmax(yhat[0, -1, :])
                char = target_dict_reverse[char_index]
                output += char
                state = [h, c]  # 本次状态做为下一次的初始状态继续传递
                predict_seq = np.zeros((1, 1, features))
                predict_seq[0, 0, char_index] = 1
                if char == '\n':  # 预测到了终止符则停下来
                    break
        return output

    def do_predict(self, input_str):
        # 去掉标点符号
        input_str = input_str.replace(',', '')
        input_str = input_str.replace('.', '')
        input_str = input_str.replace('!', '')
        input_str = input_str.replace('?', '')
        input_str = input_str.replace('，', '')
        input_str = input_str.replace('。', '')
        input_str = input_str.replace('！', '')
        input_str = input_str.replace('？', '')
        if bool(re.search('[a-zA-Z]', input_str)):  # 英文翻译为中文
            if len(input_str) > self.en2zh_INUPT_LENGTH:
                return '输入太长，请重新输入'
            input_str = input_str.lower()
            test = np.zeros((1, self.en2zh_encoder_input.shape[1], self.en2zh_encoder_input.shape[2]))
            for char_index, char in enumerate(input_str):
                test[0, char_index, self.en2zh_input_dict[char]] = 1
            out = self.predict(test, self.en2zh_encoder_infer, self.en2zh_decoder_infer,
                               self.en2zh_decoder_output.shape[1],
                               self.en2zh_decoder_output.shape[2], self.en2zh_target_dict,
                               self.en2zh_target_dict_reverse)

        else:  # 中文翻译为英文
            if len(input_str) > self.zh2en_INUPT_LENGTH:
                return ('输入太长，请重新输入')
            test = np.zeros((1, self.zh2en_encoder_input.shape[1], self.zh2en_encoder_input.shape[2]))
            for char_index, char in enumerate(input_str):
                test[0, char_index, self.zh2en_input_dict[char]] = 1
            out = self.predict(test, self.zh2en_encoder_infer, self.zh2en_decoder_infer,
                               self.zh2en_decoder_output.shape[1],
                               self.zh2en_decoder_output.shape[2], self.zh2en_target_dict,
                               self.zh2en_target_dict_reverse)
        return out


# for i in range(1,10):
#     test = encoder_input[i:i+1,:,:]#i:i+1保持数组是三维
#     out = predict_chinese(test,encoder_infer,decoder_infer,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH)
#     #print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
#     print(input_texts[i])
#     print(out)

# webapp
app = Flask(__name__, static_folder='templates', static_url_path='')

'''
request.json
一维数组，784个特征
[255, 161, 0, 0,....]
'''


@app.route('/api/chat', methods=['GET'])
def chat():
    input_str = request.args.get('message')

    print(input_str)
    # 一维数组，输出10个预测概率
    out = translateModel.do_predict(input_str)
    # 判断输入是中文还是英文，英文翻译为中文，中文翻译为英文
    return out
    # return "你也好"


@app.route('/')
def main():
    return render_template('index.html')
    # return 'Hello World!'


if __name__ == '__main__':
    translateModel = TranslateModel()
    app.run(host='0.0.0.0', port=8001)
    # app.run(host='localhost',port=8889)
    # print(loadModel.predict("你好"))
    # print(loadModel.predict("我喜欢你"))

# while True:
#     tip = '请输入英文或者中文，自动识别并翻译：'
#     input_str = input(tip)
#     if input_str is None or input_str.strip() == '':
#         continue
#     if input_str == r'\b':
#         print('再见！')
#         exit()
#     input_str = input_str.strip()
#     # print(q)
#     # list =
#
#     # 判断输入是中文还是英文，英文翻译为中文，中文翻译为英文
#     if bool(re.search('[a-zA-Z]', input_str)):
#         if len(input_str)>en2zh_INUPT_LENGTH:
#             print('输入太长，请重新输入')
#             continue
#         test = np.zeros((1, en2zh_encoder_input.shape[1], en2zh_encoder_input.shape[2]))
#         for char_index, char in enumerate(input_str):
#             test[0, char_index, en2zh_input_dict[char]] = 1
#         out = predict_chinese(test, en2zh_encoder_infer, en2zh_decoder_infer, en2zh_decoder_output.shape[1], en2zh_decoder_output.shape[2], en2zh_target_dict, en2zh_target_dict_reverse)
#         #print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
#         # print(input_texts[i])
#         print(out)
#     else:
#         if len(input_str)>zh2en_INUPT_LENGTH:
#             print('输入太长，请重新输入')
#             continue
#         test = np.zeros((1, zh2en_encoder_input.shape[1], zh2en_encoder_input.shape[2]))
#         for char_index, char in enumerate(input_str):
#             test[0, char_index, zh2en_input_dict[char]] = 1
#         out = predict_chinese(test, zh2en_encoder_infer, zh2en_decoder_infer, zh2en_decoder_output.shape[1], zh2en_decoder_output.shape[2], zh2en_target_dict, zh2en_target_dict_reverse)
#         #print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
#         # print(input_texts[i])
#         print(out)
