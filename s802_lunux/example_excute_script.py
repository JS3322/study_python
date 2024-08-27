# pip install pexpect

import pexpect

def input_id(aaa):
    # 리눅스 터미널에서 id를 입력받는 명령어를 실행합니다.
    child = pexpect.spawn('bash')  # 터미널 시작
    child.sendline('id')  # id 명령 실행

    # id를 입력하라는 프롬프트가 나타날 때까지 기다립니다.
    child.expect('id')

    # 변수 aaa의 값을 입력합니다.
    child.sendline(aaa)

    # 결과를 출력합니다.
    child.interact()  # 터미널 상호작용을 계속합니다.
