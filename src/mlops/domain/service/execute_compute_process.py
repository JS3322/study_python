

# hpc 서버에서 project 할당 갯수만큼 비동기 순서 작업 방식
@execution_time_decorator
async def execute_compute_process():
    tasks = [compute_square(i) for i in range(1, 21)]
    results = await asyncio.gather(*tasks)

@execution_time_decorator
async def compute_square(x):
    result = x * x
    await asyncio.sleep(1)
    run_loop_info = asyncio.get_running_loop()
    get_loop_info = asyncio.get_event_loop()
    return result
  
# hpc 서버에서 project 할당 갯수만큼 일괄 작업 방식
def long_time_task_example():
    data = range(100000)
    num_cores = cpu_count()
    chunks = [data[i:4] for i in range(4)]
    with Pool(4) as p:
    	results = p.map(my_task_use_process, chunks)
        
def my_task_use_process(data_chunk):
    result = sum(data_chunk)
    return result