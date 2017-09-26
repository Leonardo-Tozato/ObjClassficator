import numpy
from keras.preprocessing import image
from keras.applications import Xception, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import os

model = Xception(weights = 'imagenet')

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
def captureImg():
    #roda o comando que captura a foto, modificar pro comando do rasp.
    os.system("fswebcam -r 299x299 --jpeg 85 -S 20 teste.jpg")
    return Image.open("teste.jpg")
    

img = captureImg()
pred = predict(model, img, (299, 299))

#Printa as top probabilidades no terminal, não sei ainda exatamente o que faremos com a saída da rede então ta ai.
for (i, (image_id, label, prob)) in enumerate(pred):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
