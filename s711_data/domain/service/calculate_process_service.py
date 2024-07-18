import asyncio
import random
from multiprocessing import Pool, cpu_count
import threading

async def execute_compute_process():
    
    tasks = [compute_square(i) for i in range(1,21)]
    # asyncio.gather(*tasks)는 주어진 비동기 작업을 리스트 병렬로 실행한 것을 모든 작업이 완료될 때까지 기다림
    # 각 작업이 완료 된 후 gather에 순서대로 반환하여 순서 보장
    # * : 여러개의 인자를 함수에 전달하는 표현식이며, tasks의 배열인자를 분해해서 전달
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")
    
async def compute_square(x):
    
    print(f"Compute square of {x}")
    result = x * x
    # 1초 대기하지만 어떤 작업이 먼저 완료될지는 이벤트 루프 스케쥴링에 따라 다름
    await asyncio.sleep(1)
    print(f"square of {x} is {result}")
    return result
    
# yield = 비동기 return 값
def generator_example():
    print('generator start')
    yield 1
    print('yielded 1')
    yield 2
    print('yielded 2')
    
# 멀티 프로세싱 자원 할당 예시
def long_time_task_example():
    data = range(100000)
    num_cores = cpu_count()
    chunks = [data:[i:4] for i in range(4)]
    # 4개의 프로세스 풀 생성 :: 시스템에 몇 개의 cpu 코어가 있는 경우 Pool(4)에 의해 생성됨
    # 각 프로세스는 cup 코어 중 하나를 사용 (process 갯수가 적으면 각 cup코어에 미 할당)
    with Pool(4) as p:
    	results = p.map(my_task_use_process, chunks)
        
def my_task_use_process(data_chunk):
    # 복잡 계산 example
    result = sum(data_chunk)
    return result
    
# 멀티 쓰레딩 자원 할당 예시 (python은 싱글 스레드(GIL)이므로 간략 코드에 의미 없음
def long_time_task_example():
    thread = []
    for i in range(4):
        # 전체 범위를 4개의 부분으로 나누어 각 스레드에 할당
        t = threading.Thread(target=my_task_use_thread(), args=(250000*i, 250000*(i+1)))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        
def my_task_use_thread(start, end):
    # 범위 내 숫자 합계 계산
    result = sum(range(start, end))
    print(f'Result: {result}')
    return result
    
    
import asyncio
import random

# 작업을 시뮬레이션하는 함수
async def task(name, duration):
    print(f"Task {name} started, will take {duration:.2f} seconds")
    await asyncio.sleep(duration)
    print(f"Task {name} completed")

# 제한 시간 내에 작업을 실행하거나 재스케줄링하는 함수
async def execute_with_timeout(task_func, name, duration, timeout, queue):
    try:
        await asyncio.wait_for(task_func(name, duration), timeout)
    except asyncio.TimeoutError:
        print(f"Task {name} exceeded timeout of {timeout} seconds and was rescheduled")
        # 후순위로 작업을 큐에 추가
        await queue.put((3, task_func, name, duration))

# 작업을 큐에 추가하는 함수
async def add_tasks_to_queue(queue):
    for i in range(10):
        duration = random.uniform(0.5, 5.0)
        priority = 1 if duration <= 2 else 2  # 간단한 우선순위 설정
        await queue.put((priority, task, f"Task-{i+1}", duration))

# 큐에서 작업을 처리하는 함수
async def process_tasks(queue, timeout):
    while not queue.empty():
        priority, task_func, task_name, duration = await queue.get()
        await execute_with_timeout(task_func, task_name, duration, timeout, queue)
        queue.task_done()

# 메인 함수
async def main():
    queue = asyncio.PriorityQueue()
    await add_tasks_to_queue(queue)
    await process_tasks(queue, timeout=2)

if __name__ == "__main__":
    asyncio.run(main())