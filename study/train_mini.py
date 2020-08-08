from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
import os
import numpy
import cv2
import pandas
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# 顔画像ファイル名から年齢を抽出
def filename2age(filename):
  tempstr = filename.replace('.jpg', '')
  tempstrs = tempstr.split('_')
  return(float(tempstrs[2]))

# ファイル名のリストからデータをロード
def load_data(dirname, filenames, img_width, img_height):
  x = []
  y = []
  for filename in filenames:
    image = cv2.imread(dirname + filename)
    image = cv2.resize(image, (img_width, img_height))
    age = filename2age(filename)
    x.append(image)
    y.append(age)
  x = numpy.array(x)
  x = x.astype('float32') / 255.
  y = numpy.array(y)
  return(x, y)

# メイン処理
if __name__ == '__main__':
  img_width, img_height = 128, 128
  if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
  else:
    input_shape = (img_width, img_height, 3)
  
  # モデル構築
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
  
  # 学習(ミニバッチ)
  
  filenames = numpy.array(os.listdir('data/')) # 102 files
  filenames_test = numpy.array(os.listdir('test_data/'))
  batch_size = 34
  n_batch_loop = 3 # 102 / 34
  epochs = 100
  loss_temp = []
  loss = []
  mae_temp = []
  mae = []
  val_loss_temp = []
  val_loss = []
  val_mae_temp = []
  val_mae = []
  for e in range(epochs):
    print('Epoch', e)
    indices = numpy.repeat(range(n_batch_loop), batch_size)
    numpy.random.shuffle(indices)
    print('indices=',indices)
    for i_batch_loop in range(n_batch_loop):
      filenames_temp = filenames[indices == i_batch_loop]
      x, y = load_data('data/', filenames_temp, img_width, img_height)
      x_test, y_test = load_data('test_data/', filenames_test, img_width, img_height)
      hist = model.fit(x, y, epochs=1, verbose=1, batch_size=batch_size, validation_data=(x_test, y_test))
      val_loss_temp.append(hist.history['val_loss'][0])
      val_mae_temp.append(hist.history['val_mean_absolute_error'][0])
      score_temp = model.evaluate(x, y, verbose=0)
      loss_temp.append(score_temp[0])
      mae_temp.append(score_temp[1])
      if i_batch_loop == 2:
        loss.append(sum(loss_temp) / 3)
        loss_temp.clear()
        val_loss.append(sum(val_loss_temp) / 3)
        val_loss_temp.clear()
        mae.append(sum(mae_temp) / 3)
        mae_temp.clear()
        val_mae.append(sum(val_mae_temp) / 3)
        val_mae_temp.clear()
  

  # 学習結果の確認 (訓練データ)
  x, y = load_data('data/', filenames, img_width, img_height)
  result = pandas.DataFrame(filenames, columns = ['filename'])
  result['actual'] = y
  result = pandas.concat([result,
    pandas.DataFrame(model.predict(x), columns = ['predict'])], axis=1)
  result.to_csv('train.csv', index=False, encoding='utf-8')
  
  # 学習結果の確認 (テストデータ)
  filenames = numpy.array(os.listdir('test_data/'))
  x, y = load_data('test_data/', filenames, img_width, img_height)
  result = pandas.DataFrame(filenames, columns = ['filename'])
  result['actual'] = y
  result = pandas.concat([result,
    pandas.DataFrame(model.predict(x), columns = ['predict'])], axis=1)
  result.to_csv('test.csv', index=False, encoding='utf-8')


  #テストデータのmseとmae
  x, y = load_data('test_data/', filenames, img_width, img_height)
  print("[mse , mae]")
  print(model.evaluate(x, y, verbose=1))



  #グラフ
  list_temp = list(range(epochs))

  plt.figure(1)
  plt.plot(list_temp, loss, label="mse")
  #plt.plot(list_temp, val_loss, label="val_mse")
  plt.title("mean_squared_error")
  plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
  plt.savefig('mse.png')

  plt.figure(2)
  plt.plot(list_temp, mae, label="mae")
  #plt.plot(list_temp, val_mae, label="val_mae")
  plt.title("mean_absolute_error")
  plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
  plt.savefig('mae.png')
  
  #保存
  model.save("mini_model.h5")