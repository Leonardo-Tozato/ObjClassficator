#coding: utf-8
import numpy
import serial
import threading
from socket import *
'''from keras.preprocessing import image
from keras.applications import Xception, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import os'''

socket_num = 3001
arduin_com = serial.Serial('/dev/ttyACM0', 9600)
#model = Xception(weights = 'imagenet')

def predict(model, img, imgSize):
    #Reajusta o tamanho da imagem para o tamanho esperado caso necessario.
    if img.size != imgSize:
        img = img.resize(imgSize)
    #Converte a imagem num array tridimensional.
    x = image.img_to_array(img)
    x = numpy.expand_dims(x, axis=0)
    #Normaliza a imagem.
    x = preprocess_input(x)
    #Faz a previsao atraves da rede.
    pred = model.predict(x)
    return imagenet_utils.decode_predictions(pred, top = 3)[0]

#Captura imagem da camera.
def capture_img():
    #roda o comando que captura a foto, modificar pro comando do rasp.
    os.system("fswebcam -r 299x299 --jpeg 85 -S 20 teste.jpg")
    return Image.open("teste.jpg")

def capture_predict(img):
	img = capture_img()
	pred = predict(model, img, (299, 299))
	#Printa as top probabilidades no terminal
	for (i, (image_id, label, prob)) in enumerate(pred):
       		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

def receive(rcv_socket):
    data = rcv_socket.recv(2048)
    cmd = list(str(data.decode('utf-8')).lower())
    cmd = ''.join([c for c in cmd if c != '\x00'])
    print (cmd)
    if cmd == 'va para frente':
	arduin_com.write('1')
	arduin_com.write('4')
    elif cmd == 'vire para direita':
	arduin_com.write('2')
	arduin_com.write('2')
    elif cmd == 'vire para esquerda':
	arduin_com.write('3')
	arduin_com.write('2')
    elif cmd == 'va para tras':
	arduin_com.write('4')
	arduin_com.write('2')
    elif cmd == 'faca barulho':
        arduin_com.write('5')
	arduin_com.write('3')
    elif cmd == 'mostre todas as funcionalidades':
        arduin_com.write('1')
	arduin_com.write('2')
        arduin_com.write('2')
	arduin_com.write('2')
        arduin_com.write('3')
	arduin_com.write('2')
        arduin_com.write('4')
	arduin_com.write('2')
        arduin_com.write('5')
	arduin_com.write('3')
    rcv_socket.send('ack'.encode(encoding='utf-8', errors='ignore'))
    rcv_socket.close()

my_socket = socket(AF_INET,SOCK_STREAM)
my_socket.bind(('',socket_num))
my_socket.listen(1)

while True:
    rcv_socket, addr = my_socket.accept()
    new_thread = threading.Thread(target=receive, args=(rcv_socket,))
    new_thread.start()
my_socket.close()
