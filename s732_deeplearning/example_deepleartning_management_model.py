import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 데이터 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델 생성 함수
def create_model():
  model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 입력 이미지를 1차원 배열로 펼침
    layers.Dense(128, activation='relu'),  # 첫 번째 완전 연결층
    layers.Dropout(0.2),                  # 과적합을 방지하기 위한 드롭아웃
    layers.Dense(10, activation='softmax') # 출력층: 10개의 클래스 중 하나를 예측
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

# 모델 저장 폴더 확인 및 생성
if not os.path.exists('models'):
  os.makedirs('models')

# 현재 시간 정보 생성
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 모델 생성 및 훈련
model = create_model()
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 모델 저장
model_accuracy = max(history.history['val_accuracy'])  # 최대 검증 정확도
model_name = f'model_{now}_accuracy_{model_accuracy:.4f}.h5'
model_path = os.path.join('models', model_name)
model.save(model_path)
print(f"모델이 저장되었습니다: {model_path}")

# models 폴더에서 가장 높은 정확도를 가진 모델 불러오기
best_model_path = max(os.listdir('models'), key=lambda x: float(x.split('_accuracy_')[1].replace('.h5', '')))
loaded_model = tf.keras.models.load_model(os.path.join('models', best_model_path))
print(f"가장 높은 정확도를 가진 모델을 불러왔습니다: {best_model_path}")

# 불러온 모델 평가
test_loss, test_acc = loaded_model.evaluate(test_images, test_labels, verbose=2)
print(f'불러온 모델의 정확도: {test_acc}')