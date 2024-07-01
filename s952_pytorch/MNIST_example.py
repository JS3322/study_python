import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터 준비
# 1. 데이터셋을 텐서로 변환하고 정규화하는 변환 정의
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 2. 훈련 데이터셋 로드 및 변환 적용
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 3. 테스트 데이터셋 로드 및 변환 적용
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 4. 데이터 로더 생성 (훈련 및 테스트 데이터셋을 배치 단위로 나눔)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 첫 번째 합성곱 레이어 (입력 채널: 1, 출력 채널: 32, 커널 크기: 3x3)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        
        # 두 번째 합성곱 레이어 (입력 채널: 32, 출력 채널: 64, 커널 크기: 3x3)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # 첫 번째 드롭아웃 레이어 (드롭아웃 비율: 0.25)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 두 번째 드롭아웃 레이어 (드롭아웃 비율: 0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # 첫 번째 완전 연결 레이어 (입력 특징 수: 9216, 출력 특징 수: 128)
        self.fc1 = nn.Linear(9216, 128)
        
        # 두 번째 완전 연결 레이어 (입력 특징 수: 128, 출력 특징 수: 10)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 첫 번째 합성곱 레이어와 ReLU 활성화 함수 적용
        x = self.conv1(x)
        x = F.relu(x)
        
        # 두 번째 합성곱 레이어와 ReLU 활성화 함수 적용
        x = self.conv2(x)
        x = F.relu(x)
        
        # 최대 풀링 레이어 적용 (2x2 풀링 창)
        x = F.max_pool2d(x, 2)
        
        # 첫 번째 드롭아웃 레이어 적용
        x = self.dropout1(x)
        
        # 특징 맵을 1차원 벡터로 평탄화
        x = torch.flatten(x, 1)
        
        # 첫 번째 완전 연결 레이어와 ReLU 활성화 함수 적용
        x = self.fc1(x)
        x = F.relu(x)
        
        # 두 번째 드롭아웃 레이어 적용
        x = self.dropout2(x)
        
        # 두 번째 완전 연결 레이어 적용
        x = self.fc2(x)
        
        # 로그 소프트맥스 활성화 함수 적용
        output = F.log_softmax(x, dim=1)
        return output

# 모델, 손실 함수, 최적화 설정
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 모델 학습
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 이전 미니 배치의 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파: 모델에 데이터를 입력하여 예측 값 계산
        output = model(data)
        
        # 손실 계산
        loss = criterion(output, target)
        
        # 역전파: 기울기 계산
        loss.backward()
        
        # 가중치 업데이트
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 모델 평가
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

# 학습 및 평가 반복
for epoch in range(1, 11):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)