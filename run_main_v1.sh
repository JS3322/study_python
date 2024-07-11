# 1. 권한 부여 
# chmod +x ~/project/fast_api_example/run_app.sh
# 2. 파일 실행
# ~/project/fast_api_example/run_app.sh

#!/bin/bash

# Python 3.10이 설치되어 있는지 확인하고, 설치되지 않은 경우 설치
if ! command -v python3.10 &> /dev/null
then
    echo "Python 3.10 not found. Installing Python 3.10..."
    sudo apt update
    sudo apt install -y python3.10
fi

# 프로젝트 디렉토리로 이동
cd ~/project/fast_api_example/

# 가상 환경이 존재하지 않으면 생성
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 가상 환경 활성화
source venv/bin/activate

# requirements.txt로 패키지 설치
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, installing default packages..."
    pip install fastapi uvicorn
fi

# FastAPI 앱 실행
python main_v1.py