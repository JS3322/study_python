/your_project
│
├── app
│   ├── main.py          # FastAPI 애플리케이션 엔트리 포인트
│   ├── dependencies.py  # 의존성 관리 파일
│   ├── api              # 애플리케이션 엔드포인트
│   │   └── api_v1       # API 버전 관리
│   │       └── endpoints
│   │           └── data.py
│   ├── domain           # 도메인 모델
│   │   └── models.py    # 엔터티 정의
│   ├── application      # 애플리케이션 비즈니스 로직
│   │   └── services.py  # 서비스 로직
│   └── adapters         # 어댑터
│       └── repository.py# 데이터베이스 리포지토리
│
└── requirements.txt     # 필요한 라이브러리













