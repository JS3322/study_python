import functools

def print_variables(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    # 함수가 실행되기 전에 받은 인자들을 출력
    print(f"Arguments: args={args}, kwargs={kwargs}")

    # 함수 실행
    result = func(*args, **kwargs)

    # 함수 내의 로컬 변수들을 출력
    print(f"Local variables: {locals()}")

    return result

  return wrapper

# 예시 함수
@print_variables
def example_function(a, b):
  c = a + b
  d = c * 2
  return d

# 함수 호출
example_function(2, 3)
