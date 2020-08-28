import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from core.features import FeatureMetas
from models.AFM import AFM

from tensorflow.python.client import device_lib
import keras.backend.tensorflow_backend as KTF
from keras.backend.tensorflow_backend import set_session

import os
if __name__ == '__main__':
      os.environ["CUDA_VISIBLE_DEVICES"] = "0"
      config = tf.compat.v1.ConfigProto()
      config.gpu_options.per_process_gpu_memory_fraction = 0.3
      session = tf.compat.v1.Session(config=config)
      # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
      # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
      # print(gpus, cpus)
      # tf.config.experimental.set_visible_devices(devices=gpus[2:4], device_type='GPU')
      #

      data = pd.read_csv('../datasets/avazu_1w.txt')
      # data = pd.read_csv('../datasets/train1.csv')


      print("GPU Available: ", tf.test.is_gpu_available())
      sparse_features = list(data.columns)
      target = ['click']

      data[sparse_features] = data[sparse_features].fillna('-1',)
      for feat in sparse_features:
          lbe = LabelEncoder()
          data[feat] = lbe.fit_transform(data[feat])

      trian , test = train_test_split(data, test_size=0.2)
      trian_0 = trian[trian.click == 0]
      trian_1 = trian[trian.click == 1]

      trian = pd.concat([trian_1,trian_0[0:len(trian_1)]])

      trian_model_input = {name : trian[name] for name in sparse_features}
      test_model_input = {name : test[name] for name in sparse_features}

      feature_metas = FeatureMetas()
      for feat in sparse_features:
            feature_metas.add_sparse_feature(name=feat, one_hot_dim=data[feat].nunique(), embedding_dim=32)

      model = AFM(
             feature_metas=feature_metas,
             linear_slots=sparse_features,
             fm_slots=sparse_features
       )
      model.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=['binary_crossentropy'])

      history = model.fit(x=trian_model_input,
                          y=trian[target].values,
                          batch_size=32,
                          epochs=1,
                          verbose=0,
                          validation_split=0.2
                          )
      print("afm")
      pred_ans = model.predict(test_model_input,batch_size=64)
      # tf.keras.utils.plot_model(model,tofile='afm.png')
      print("test LogLoss",round(log_loss(test[target].values,pred_ans),4))
      print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))







