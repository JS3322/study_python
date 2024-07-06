
- image build
```cmd
docker build -t study-python-local-env:0.0.1 -f docker/dockerfile_local .
```

- create container
```cmd
docker run -it --rm -v $(pwd):/app -p 8000:8000 --name study-python-container study-python-local-env:0.0.1
```

- start container
```
docker start study-python-container
```

- delete container
```
docker ps
docker stop study-python-container
docker rm study-python-container
```

- delete image
```
docker rmi study-python-dev-env
```

- 인텔리제이는 image에 repo를 마운트하여 실행