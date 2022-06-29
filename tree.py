import pandas as pd
import numpy as np
from sklearn import tree
from pprint import pprint

import graphviz


def get_data():
    data = pd.read_excel('./data.xlsx', engine='openpyxl')
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




if __name__ == '__main__':
    df = get_data()
    df_data = df.drop(['label'], axis=1)
    df_label = df['label']
    # pprint(df_label)
    train_data = df_data.to_numpy()
    train_label = df_label.to_numpy()

    # print(train_data.shape)
    # print(train_label.shape)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_label)

    res = clf.predict(train_data)
    pprint(res)

    # # 绘制一下决策树
    dot_data = tree.export_graphviz(
        clf,
        out_file='tree.dot',
        feature_names=df_data.columns.to_numpy(),
        class_names=df['label'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    #
    # graph = graphviz.Source(dot_data)
    # #
    # graph.render(directory='./doctest-output', view=True)

