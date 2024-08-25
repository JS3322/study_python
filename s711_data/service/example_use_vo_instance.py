from dataclasses import dataclass

# VO 클래스 정의
@dataclass
class UserVO:
  name: str
  age: int
  email: str

# 메인 클래스
class User:
  def __init__(self, user_vo: UserVO):
    # VO 객체를 인스턴스 변수로 설정
    self.vo = user_vo

  def display_info(self):
    return f"Name: {self.vo.name}, Age: {self.vo.age}, Email: {self.vo.email}"

# 사용 예시
# VO 객체 생성
user_vo = UserVO(name="John Doe", age=30, email="johndoe@example.com")

# VO 객체를 통해 User 인스턴스 생성
user = User(user_vo)

# 인스턴스 정보 출력
print(user.display_info())
