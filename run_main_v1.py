import os
import subprocess
import sys

# 프로젝트 디렉토리로 이동
project_dir = os.path.expanduser("~/project/fast_api_example")
os.chdir(project_dir)

# 가상 환경 디렉토리
venv_dir = os.path.join(project_dir, "venv")

# 가상 환경이 존재하지 않으면 생성
if not os.path.isdir(venv_dir):
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", "venv"])

# 가상 환경 활성화
activate_script = os.path.join(venv_dir, "bin", "activate_this.py")
with open(activate_script) as file_:
    exec(file_.read(), dict(__file__=activate_script))

# requirements.txt로 패키지 설치
requirements_file = os.path.join(project_dir, "requirements.txt")
if os.path.isfile(requirements_file):
    print("Installing requirements...")
    subprocess.check_call([os.path.join(venv_dir, "bin", "pip"), "install", "-r", "requirements.txt"])
else:
    print("requirements.txt not found, installing default packages...")
    subprocess.check_call([os.path.join(venv_dir, "bin", "pip"), "install", "fastapi", "uvicorn"])

# FastAPI 앱 실행
subprocess.check_call([os.path.join(venv_dir, "bin", "python"), "main_v1.py"])