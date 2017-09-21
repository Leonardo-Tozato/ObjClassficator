import sys
import argparse
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.applications import Xception
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
model = Xception(weights = 'imagenet')

def predict(model, img, imgSize):
    """ Essa funcao faz o reconhecimento dos objetos utilizando o modelo carregado.
    Argumentos:
     - model: modelo previamente carregado para a predicao.
     - img: imagem no formato PIL.
     - imgSize: tamanho esperado da imagem.
    Retorno: lista com as 5 labels mais provaveis preditas. """
    
    #Reajusta o tamanho da imagem para o tamanho esperado caso necessario.
    if img.size != imgSize :
        img = img.resize(imgSize)

    #Converte a imagem num array tridimensional.
    x = image.img_to_array(img)
    x = numpy.expand_dims(x, axis=0)
    #Normaliza a imagem.
    x = preprocess_input(x)
    
    #Faz a previsao atraves da rede.
    pred = model.predict(x)
    return imagenet_utils.decode_predictions(pred, top=5)[0]
    
def plotPred(img, pred):
    """Produz um grafico de barras com os top5 labels preditos.
    Argumentos:
    - img: imagem que teve o objeto reconhecido.
    - pred: vetor com top5 labels eleitos pela CNN.
    """

    #plota a imagem.
    plt.imshow(img)
    plt.axis('off')

    #grafico de barras.
    plt.figure()  
    order = list(reversed(range(len(pred))))  
    bar_preds = [pr[2] for pr in pred]
    labels = (pr[1] for pr in pred)
    plt.barh(order, bar_preds, alpha=0.5)
    plt.yticks(order, labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()

"""Main do programa que toma uma imagem do sistema de arquivos, 
chama o classificador e em seguida plota a imagem"""
if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  args = a.parse_args()

  if args.image is None:
    a.print_help()
    sys.exit(1)

  if args.image is not None:
    img = Image.open(args.image)
    pred = predict(model, img, (299, 299))
    plotPred(img, pred)