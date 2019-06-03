from  socket import *

print("채팅 프로그램입니다.\n")

host = '127.0.0.1' #localhost
port = 8080
server = socket(AF_INET, SOCK_STREAM)
server.bind((host,port))
server.listen(1)

conn, addr = server.accept()

print(str(addr),'에서 접속이 확인 되었습니다.')

while True :
    send = input('나  : ')
    conn.send(send.encode('utf-8'))

    recv = conn.recv(1024)
    print('상대방 :', recv.decode('utf-8'))
