import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# load data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# crate model
model = models.Sequential([
  layers.Flatten(input_shape=(28, 28)),  # 입력 이미지를 1차원 배열로 펼침
  layers.Dense(128, activation='relu'),  # 첫 번째 완전 연결층
  layers.Dropout(0.2),                  # 과적합 방지 드롭아웃
  layers.Dense(10, activation='softmax') # 출력층: 10개의 클래스 중 하나를 예측
])

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# save model
model.save('mnist_model.h5')
print("save model")

# graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.show()

# model test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# load model
loaded_model = tf.keras.models.load_model('mnist_model.h5')

# load model test
loaded_model_test_loss, loaded_model_test_acc = loaded_model.evaluate(test_images, test_labels, verbose=2)
print(f'load model Test accuracy: {loaded_model_test_acc}')