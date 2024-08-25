import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 데이터 생성 및 증설
def doe_data_augmentation(X, y, factor=2):
  augmented_X = np.tile(X, (factor, 1))  # 데이터를 증설 (factor 배)
  augmented_y = np.tile(y, factor)  # 레이블 증설
  # 예시로 노이즈를 추가
  noise = np.random.normal(0, 0.1, augmented_X.shape)
  augmented_X = augmented_X + noise
  return augmented_X, augmented_y

# MLflow 실험 시작
mlflow.start_run()

# 데이터 생성
X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
X_augmented, y_augmented = doe_data_augmentation(X, y, factor=5)  # 데이터 증설

# 데이터 전처리
scaler = StandardScaler()
X_augmented = scaler.fit_transform(X_augmented)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# MLflow에 메트릭 기록
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "random_forest_model")

# 실험 종료
mlflow.end_run()

print(f"Model accuracy: {accuracy}")
