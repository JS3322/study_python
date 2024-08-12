# 모듈 로드
module load python/3.8

# 가상환경 생성
python -m venv ~/envs/mlops_env
source ~/envs/mlops_env/bin/activate

#필요한 라이브러리 설치
pip install mlflow dvc pytorch-lightning