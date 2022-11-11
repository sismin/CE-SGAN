import math

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from sklearn import preprocessing
import datetime

# 加载美国油田数据集
filename = './Dataset/美国油田/train_data.csv'
filename_blind = './Dataset/美国油田/blind_data.csv'
train_data = pd.read_csv(filename)
test_data = pd.read_csv(filename_blind)
train_data = train_data.dropna()
test_data = test_data.dropna()

all_data = pd.concat([train_data,test_data],axis=0)
x_all_data = all_data.drop(columns=['Formation', 'Well Name', 'Depth','Facies'])
y_all_data = all_data['Facies']-1
max_min = preprocessing.StandardScaler()
x_all_data = max_min.fit_transform(x_all_data)
x_train = x_all_data[:-830,]
y_train = y_all_data[:-830]
x_test = x_all_data[-830:,]
y_test = y_all_data[-830:]
x_train = x_train.reshape(3232,7,1)
x_test = x_test.reshape(830,7,1)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=4000)
train_dataset = train_dataset.batch(300)
latent_dim = 4
ti = math.ceil(len(x_train)/300)
tf.keras.backend.set_floatx('float64')

# 分类器
def Classifier():
  model = keras.models.Sequential()
  model.add(keras.layers.Conv1D(64,(2,),1,padding='same',input_shape=(7,1),activation='relu'))
  model.add(keras.layers.Conv1D(64,(2,),1,padding='same',activation='relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.MaxPooling1D(2,2))
  model.add(keras.layers.Conv1D(128,(2,),1,padding='same',activation='relu'))
  model.add(keras.layers.Conv1D(128,(2,),1,padding='same',activation='relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.MaxPooling1D(3,1))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(2500,activation='relu'))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(1500,activation='relu'))
  model.add(keras.layers.Dense(9,activation='softmax'))
  return model

# 生成器
def Generator():
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(latent_dim,)))
  model.add(layers.Reshape((1,latent_dim)))
  model.add(layers.Dense(16))
  model.add(layers.Dense(64))
  model.add(layers.LeakyReLU())
  model.add(layers.Dense(16))
  model.add(layers.LeakyReLU())
  model.add(layers.Dense(7,activation='tanh'))
  model.add(layers.Reshape((7,1)))
  # model.summary()
  return model

# 判别器
def Discriminator():
  model = tf.keras.Sequential()
  model.add(layers.Flatten())
  model.add(layers.Dense(64))
  model.add(layers.Dense(128,activation='relu'))
  model.add(layers.Dense(64))
  model.add(layers.Dense(1,activation='sigmoid'))
  # model.summary()
  return model

#初始化
generatorLosses = []
generatorAcc = []
discriminatorLosses = []
classifierLosses = []
discriminatorAcc = []
discriminatorAcc_real = []
discriminatorAcc_fake = []
classifierAcc = []
eval_acc = []
test_acc = []
# CategoricalAccuracy = tf.keras.metrics.CategoricalAccuracy()
CategoricalAccuracy = tf.keras.metrics.SparseCategoricalAccuracy()
BinaryAccuracy = tf.keras.metrics.BinaryAccuracy()
discriminator = Discriminator()
generator = Generator()
classifier = Classifier()
# 优化器设置
optD = tf.keras.optimizers.SGD(learning_rate=0.0002)
optG = tf.keras.optimizers.SGD(learning_rate=0.0002)
optC = tf.keras.optimizers.Adam(learning_rate=0.0001)
advWeight = 0.1  # adversarial weight
# discriminator.compile(loss='binary_crossentropy' ,optimizer=optD ,metrics=['accuracy'])
# classifier.compile(loss='categorical_crossentropy',optimizer=optC ,metrics=['accuracy'])

noise = tf.keras.Input(shape=(latent_dim,))
gen_sample = generator(noise)
predlabels = discriminator(gen_sample)
generator_ = Model(noise, predlabels)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# 定义模型保存的函数
checkpoint = tf.train.Checkpoint(model=classifier)

def Train(epochs):
  for epoch in range(epochs):
    # count = 1
    maxacc = 0.0
    print("第"+str(epoch+1)+"个epoch：")
    for batch_x,batch_y in train_dataset:
      # count = count+1
      a = ''
      batch_size = len(batch_x)
      true_label = np.ones((batch_size, 1))
      fake_label = np.zeros((batch_size, 1))
      noise = np.random.normal(0, 1, (batch_size,latent_dim))
      gen_samples = generator(noise)
      discriminator.trainable = True
      #在真实样本中 训练 discriminator
      with tf.GradientTape() as dis_tape:
        real_output = discriminator(batch_x)
        dis_loss1 = tf.keras.losses.BinaryCrossentropy()(true_label,real_output)
      gradients_of_dis = dis_tape.gradient(dis_loss1,discriminator.trainable_variables)
      optD.apply_gradients(zip(gradients_of_dis,discriminator.trainable_variables))

      #在生成样本中 训练 discriminator
      with tf.GradientTape() as dis_tape:
        fake_output = discriminator(gen_samples)
        dis_loss2 = tf.keras.losses.BinaryCrossentropy()(fake_label,fake_output)
      gradients_of_dis = dis_tape.gradient(dis_loss2,discriminator.trainable_variables)
      optD.apply_gradients(zip(gradients_of_dis,discriminator.trainable_variables))

      # 训练 generator
      discriminator.trainable = False
      with tf.GradientTape() as gen_tape:
        out_label = generator_(noise)
        gen_loss = tf.keras.losses.BinaryCrossentropy()(true_label,out_label)
      gradients_of_gen = gen_tape.gradient(gen_loss,generator_.trainable_variables)
      optG.apply_gradients(zip(gradients_of_gen,generator_.trainable_variables))

      #在真实样本中 训练 classifier
      with tf.GradientTape() as class_tape:
        pred_label = classifier(batch_x)
        class_loss = tf.keras.losses.SparseCategoricalCrossentropy()(batch_y,pred_label)
        CategoricalAccuracy.update_state(batch_y,pred_label)
        class_acc = CategoricalAccuracy.result().numpy()
        CategoricalAccuracy.reset_states()
      gradients_of_classifier = class_tape.gradient(class_loss,classifier.trainable_variables)
      optC.apply_gradients(zip(gradients_of_classifier,classifier.trainable_variables))

      #筛选生成样本并 训练 classifier
      with tf.GradientTape() as class_tape:
        predictionsFake = classifier(gen_samples)
        predictedLabels = tf.math.argmax(predictionsFake, 1)
        predictedLabels = tf.cast(predictedLabels,dtype=tf.float32)
        mostLikelyProbs = np.max(predictionsFake,axis=1)
        confidenceThresh = 0.3
        toKeep = mostLikelyProbs > confidenceThresh
        # advWeight
        classifier_loss = tf.keras.losses.SparseCategoricalCrossentropy()(predictedLabels[toKeep],predictionsFake[toKeep]) * CategoricalAccuracy.result().numpy() * 0.5
      gradients_of_classifier = class_tape.gradient(classifier_loss, classifier.trainable_variables)
      optC.apply_gradients(zip(gradients_of_classifier,classifier.trainable_variables))

      y_pred = classifier(x_test)
      CategoricalAccuracy.update_state(y_test,y_pred)
      acc = CategoricalAccuracy.result().numpy()
      CategoricalAccuracy.reset_states()
      if maxacc<=acc:
        classifier.save('best_model.h5')
        maxacc = acc
      #计算每一个批次的训练损失和准确率
      dis_loss = dis_loss1+dis_loss2
      discriminatorLosses.append(dis_loss)
      generatorLosses.append(gen_loss)
      classifierLosses.append(class_loss)
      a += str(dis_loss.numpy())+'  '+str(gen_loss.numpy())+'  '+str(class_loss.numpy())+'  '
      classifierAcc.append(class_acc)
      a += ' classifierAcc '+str(class_acc)
      #测试盲井准确率
      eval_acc.append(maxacc)
      a += ' eval_acc: '+str(maxacc)
      print(a)
      checkpoint.save(file_prefix=r'save_check/logs')
      with train_summary_writer.as_default():
        tf.summary.scalar('discriminatorLosses', dis_loss, step=epoch)
        tf.summary.scalar('generatorLosses', gen_loss, step=epoch)
        tf.summary.scalar('classifierLosses', class_loss, step=epoch)
        tf.summary.scalar('classifierAcc', class_acc, step=epoch)
      with test_summary_writer.as_default():
        tf.summary.scalar('eval_acc', maxacc, step=epoch)

if __name__ == '__main__':
  tf.keras.backend.set_floatx('float64')
  Train(5000)