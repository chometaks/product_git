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
  
    #学習
    filenames = numpy.array(os.listdir('data/')) # 102 files
    batch_size = 102
    epochs = 100
    x, y = load_data('data/', filenames, img_width, img_height)
    filenames_test = numpy.array(os.listdir('test_data/'))
    x_test, y_test = load_data('test_data/', filenames_test, img_width, img_height)
    hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))


    # 学習結果の確認 (訓練データ)
    x, y = load_data('data/', filenames, img_width, img_height)
    result = pandas.DataFrame(filenames, columns = ['filename'])
    result['actual'] = y
    result = pandas.concat([result,
        pandas.DataFrame(model.predict(x), columns = ['predict'])], axis=1)
    result.to_csv('train2.csv', index=False, encoding='utf-8')
  
    # 学習結果の確認 (テストデータ)
    filenames = numpy.array(os.listdir('test_data/'))
    x, y = load_data('test_data/', filenames, img_width, img_height)
    result = pandas.DataFrame(filenames, columns = ['filename'])
    result['actual'] = y
    result = pandas.concat([result,
        pandas.DataFrame(model.predict(x), columns = ['predict'])], axis=1)
    result.to_csv('test2.csv', index=False, encoding='utf-8')


    #テストデータのmseとmae
    x, y = load_data('test_data/', filenames, img_width, img_height)
    print(model.evaluate(x, y, verbose=1))



    #グラフ
    loss = hist.history['loss']
    mae = hist.history['mean_absolute_error']

    plt.figure(1)
    plt.plot(range(epochs), loss, label="mse")
    plt.plot(range(epochs), hist.history['val_loss'], label="validation")
    plt.title("mean_squared_error")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.savefig('mse2.png')

    plt.figure(2)
    plt.plot(range(epochs), mae, label="mae")
    plt.plot(range(epochs), hist.history['val_mean_absolute_error'], label="validation")
    plt.title("mean_absolute_error")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.savefig('mae2.png')
  