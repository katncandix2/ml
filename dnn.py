import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pprint import pprint


def get_data():
    data = pd.read_excel('../data.xlsx', engine='openpyxl')
    # pprint(data)

    # pprint(data.columns)

    data = data.drop(['记录序号'], axis=1)

    data1 = data.rename(
        columns={"天气状况": "weather", "温度": "temp", "湿度": "humidity", "风力": "wind", "是否适合游玩(预测变量)": "label"})
    # pprint(data1)

    data1['weather'].unique()

    allDict = {}
    for col in data1:
        dataList = data1[col].unique()
        tmpDict = {}
        for i, v in enumerate(dataList):
            tmpDict[v] = i
            allDict[col] = tmpDict

    # pprint(allDict)

    # 文字转换成编码形式
    df1 = data1['weather'].apply(lambda x: allDict['weather'][x])
    # pprint(df1)

    for col in data1:
        dic = allDict[col]
        df1 = data1[col].apply(lambda x: dic[x])
        data1[col] = df1

    # pprint(data1)
    return data1


def save_model():
    # 1.get data
    df = get_data()
    df_data = df.drop(['label'], axis=1)
    df_label = df['label']

    # pprint(df_data)
    # pprint(df_label)
    train_data = tf.data.Dataset.from_tensor_slices(df_data.to_numpy())
    train_label = tf.data.Dataset.from_tensor_slices(df_label.to_numpy())

    model = keras.Sequential([
        # 其实这里可以换一下
        layers.Flatten(input_shape=(4, 1)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  # loss 也可以换一下思考
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    pprint(model.summary())
    model.fit(df_data.to_numpy(), df_label.to_numpy(), epochs=500)
    model.save('./dnn_model_file/')


if __name__ == '__main__':
    # 训练模型
    save_model()

    df = get_data()
    df_data = df.drop(['label'], axis=1)
    df_label = df['label']

    # load_model 验证一下
    model1 = tf.keras.models.load_model('./dnn_model_file/')
    predictions = model1.predict(df_data.to_numpy())
    pprint((predictions>=0.5).astype(int))
    pprint(df_label)
