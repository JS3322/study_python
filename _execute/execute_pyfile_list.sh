#!/bin/bash

# 파이썬 가상환경 활성화
source ../src/venv/bin/activate

# 파이썬 파일들이 있는 디렉토리로 이동
cd ../src/mlops/application/

# 현재 디렉토리의 모든 파이썬 파일을 찾음
files=$(ls *.py)

# 파일들을 하나씩 실행
for file in $files; do
  echo "Running $file..."

  # 파이썬 파일 실행
  python3 "$file"

  # 실행 결과 코드 확인
  if [ $? -eq 0 ]; then
    echo "$file executed successfully with system code 0."
  else
    echo "$file execution failed with system code 1."
  fi
done
