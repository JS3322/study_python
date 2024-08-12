import asyncio
import concurrent.futures
import subprocess

# 더미 함수로, 실제로는 리눅스 명령을 실행하는 코드로 대체해야 합니다.
def execute_job(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        return e.stderr.decode()

# 비동기 함수로, 여러 job을 병렬로 실행합니다.
async def run_jobs(commands):
    # ThreadPoolExecutor를 사용하여 비동기 작업 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        loop = asyncio.get_running_loop()
        
        # 각 명령을 별도의 스레드에서 실행하는 비동기 작업을 스케줄링
        futures = [
            loop.run_in_executor(executor, execute_job, cmd)
            for cmd in commands
        ]
        
        # 모든 비동기 작업이 완료될 때까지 기다립니다.
        results = await asyncio.gather(*futures)
        return results

# 메인 함수로, 비동기 실행을 시작합니다.
async def main():
    # 실행할 리눅스 명령어 리스트
    commands = ["ls", "whoami", "pwd"] * 20  # 예제를 위해 명령어를 반복
    results = await run_jobs(commands)
    
    for result in results:
        print(result)

# 비동기 메인 함수 실행
asyncio.run(main())