from socket import *

print("채팅 프로그램입니다.\n")

host = '127.0.0.1'
port = 8080

# 8080포트의 로컬로 연결
client = socket(AF_INET, SOCK_STREAM)
client.connect((host,port))

print("연결이 확인 되었습니다.")

while True:
    recv = client.recv(1024)
    print('상대방 :', recv.decode('utf-8'))

    send = input('나 : ')
    client.send(send.encode('utf-8'))
