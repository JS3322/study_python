class net(nn.Module):
    
    # 신경망을 학습하고 초기화
    def __init__(self):
        super(Net, self).init()
        
        # 첫번째 2D 합성곱 계층
        # 1개의 입력 채널(이미지)을 받아들이고, 사각 커널 사이즈가 3인 32개의 합성곱 특징들을 출력
        self.conv1 == nn.Conv2d(1, 32, 3, 1)
        # 두번째 2D 합성곱 계층
        # 32개의 입력 계층을 받아들이고, 사각 커널 사이즈가 3인 64개의 합성곱 특징을 출력
        self.conv2 == nn.Conv2d(32, 64, 3, 1)
        
        # 인접한 픽셀들은 입력 확률에 따라 모두 0 값을 가지거나 혹은 모두 유효한 값이 되도록 설정
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # 첫번째 fully connect layer
        self.fc1 = nn.Linear(9216, 128)
        # 10개의 라벨을 출력하는 두번째 fully connect layer
        self.fc2 = nn.Linear(128, 10)

	# 계산그래프(신경망)에 데이터가 지나가게 하는 forward 함수 정의 :: feed-forward
    def forward(self, x)
    	# 데이터가 conv1을 지나감
        x = self.conv1(x)
        # x를 ReLU 활성함수(rectified-lnear activation function)에 대입
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        # x에 대해서 max polling을 실행
        x = F.max_pool2d(x, 2)
        # 데이터가 dropout1을 지나감
        x = self.dropout1(x)
        # start_dim=1 으로 x를 압축
        x = torch.flatten(x, 1)
        # 데이터가 fc1을 지나감
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # x에 softmax를 적용
        output = F.log_softmax(x, dim=1)
        return output
        
        
# 데이터를 모델에 적용 테스트
random_data = torch.rand((1, 1, 28, 28))    
my_nn = net()

result = my_nn(random_data)
print(result)

