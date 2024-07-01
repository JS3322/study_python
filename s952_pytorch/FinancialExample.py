import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import glob

# 데이터셋 클래스 정의
class FinancialDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 라벨링 규칙에 따라 데이터셋 생성
def load_dataset(asset_folder):
    image_paths = []
    labels = []

    for label, folder in enumerate(['rich', 'likes_growth', 'other']):
        path = os.path.join(asset_folder, folder, '*.jpg')
        for img_path in glob.glob(path):
            image_paths.append(img_path)
            labels.append(label)

    return image_paths, labels

# 데이터셋 로드 및 전처리
asset_folder = 'asset'
image_paths, labels = load_dataset(asset_folder)

# 데이터 나누기
X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# 데이터 증강 및 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FinancialDataset(X_train, y_train, transform=transform)
val_dataset = FinancialDataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 정의 (pretrained ResNet 사용)
class FinancialModel(nn.Module):
    def __init__(self, num_classes):
        super(FinancialModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

num_classes = 3  # 'rich', 'likes_growth', 'other'
model = FinancialModel(num_classes=num_classes).to('cuda')

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수 정의
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        # 각 epoch마다 학습 및 검증 단계
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # 데이터 반복
            for inputs, labels in data_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 모델 복사 (deep copy)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

# 모델 학습
model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25)

# 모델 저장
torch.save(model.state_dict(), 'financial_model_recommender.pth')

# 테스트 이미지 분류 예제
def classify_image(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    return preds.item()

# 테스트 이미지 분류
test_image_path = 'path_to_test_image.jpg'
predicted_label = classify_image(model, test_image_path, transform)
label_names = ['돈이 여유로움', '돈을 불리는 것을 좋아함', '그 외']
print(f'The predicted label is: {label_names[predicted_label]}')