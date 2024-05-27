#### todo
- [ ] in-memory 전략
  - update에 target collection을 특정 인스턴스에 이관하여 활용
  - mongod --port 202111 --bpath /path/to/cleancode/data --logpath /path/to/cleancode/log/mongod.log --fork
  - mongod --port 202111 --storageEngin inMemory --dbpath /path/to/cleancode/data --logpath /path/to/hcms/log/mongod.log --fork

---
#### TEST case
```bash
uvicorn main_cleancode_v1:app --reload
```
```
http://127.0.0.1:8000/redoc
```
```
http://127.0.0.1:8000/docs
```
---
#### Sturuct
- Hexagonal
  - API 송수신 및 DB 데이터 query 관리에 용이한 구조 설계
  - mongoDB enterprise와 cleancode 서비스에 적합한 데이터 일괄 관리 구조
- Job
  - 하나의 request API는 하나의 컬렉션에 대해 job을 실행한다는 규칙
  - 하나의 response API는 하나의 데이터 수집/정제/통계 job을 실행한다는 규칙

```

cleancode - adapter									# adapter layer
celancode - application							# apllication layer
cleancode - config									# 설정정보
cleancode - domain - pipeline			# mongodb query pipeline 정보
cleancode - domain - request - job	# 작업목록
cleancode - domain - request - vo	# request body 파라미터

batch 		# migration batch repo :: 예정
docker		# 배포 시 docker yml 정보
requirements.txt	# 필요 라이브러리 관리
main.py	# 애플리케이션 엔트리 함수

``` 