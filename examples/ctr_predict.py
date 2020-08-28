import pandas as pd
import tensorflow as tf


#from keras.utils import plot_model
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from models.DeepFM import DeepFM
from models.WideAndDeep import WideAndDeep
from models.FNN import FNN
from models.PNN import PNN
from models.DCN import DCN
from models.xDeepFM import xDeepFM
from models.NFM import NFM
from models.AFM import AFM
from models.AutoInt import AutoInt
from models.CCPM import CCPM
from models.FGCNN import FGCNN
from core.features import FeatureMetas
import keras.backend.tensorflow_backend as KTF
import  os
import pydot


if __name__ == "__main__":
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
    # Read dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    data = pd.read_csv('../datasets/avazu_1w.txt')
    data = pd.read_csv('../datasets/train_500w.csv')
    # Get columns' names
    sparse_features = list(data.columns)
    target = ['click']

    # Preprocess your data 将离散数据进行编码
    data[sparse_features] = data[sparse_features].fillna('-1', )
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])



    # Split your dataset
    train, test = train_test_split(data, test_size=0.2)
    train_0 = train[train.click == 0]

    train_1 = train[train.click == 1]


    train = pd.concat([train_1, train_0[0:len(train_1)]])


    train_model_input = {name: train[name] for name in sparse_features}
    test_model_input = {name: test[name] for name in sparse_features}

    # Instantiate a FeatureMetas object, add your features' meta information to it

    feature_metas = FeatureMetas()
    for feat in sparse_features:
        feature_metas.add_sparse_feature(name=feat, one_hot_dim=data[feat].nunique(), embedding_dim=32)
    # print(len(sparse_features))
    # Instantiate a model and compile it

    model = DeepFM(
        feature_metas=feature_metas,
        linear_slots=sparse_features,
        fm_slots=sparse_features,
        dnn_slots=sparse_features
    )
    # model = AFM(
    #     feature_metas=feature_metas,
    #     linear_slots=sparse_features,
    #     fm_slots=sparse_features
    # test LogLoss 0.0246
    # test AUC 1.0
    # )

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['binary_crossentropy'])
    # print("###########################")
    # print(feature_metas.sparse_feats_slots)
    # print(feature_metas.dense_feats_slots)
    # print(feature_metas.all_feats_slots)
    # Train the model
    # verbose 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    #epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
    #print(test_model_input)

    history = model.fit(x=train_model_input,
                        y=train[target].values,
                        batch_size=128,
                        epochs=1,
                        verbose=0,
                        validation_split=0.2)

    # plot_model(model, to_file='model.png')
    # Testing
    pred_ans = model.predict(test_model_input, batch_size=50)
    tf.keras.utils.plot_model(model, to_file='model.png')
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

