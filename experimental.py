from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt

#Dicionário para mapear o argumento para a classe do modelo.
MODELS = {
    "ResNet50" : ResNet50,
    "InceptionV3": InceptionV3,
    "Xception": Xception,
    "VGG16": VGG16,
    "VGG19": VGG19
}

def predict(args) :
    #Inicializa o tamanho esperado e o pre processamento da imagem de acordo com a rede selecionada.
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

    if args["model"] in ("InceptionV3", "Xception") :
        inputShape = (299, 299)
        preprocess = preprocess_input

    #Carrega o modelo passado como argumento.
    print("Carregando modelo...")
    Network = MODELS[args["model"]]
    model = Network(weights="imagenet")

    #Carrega a imagem.
    print("Carregando e pre processando imagem...")
    img = load_img(args["image"], target_size = inputShape)
    img = img_to_array(img)

    #Adicionado dimensao de batch.
    img = np.expand_dims(img, axis=0)

    #Pre processa a imagem de acordo com o modelo selecionado.
    img = preprocess(img)

    #Chama o modelo selecionado e classifica a imagem. 
    print("Classificando a imagem através do modelo {}...".format(args["model"]))
    preds = model.predict(img)
    P = imagenet_utils.decode_predictions(preds)[0]
    plotPred(Image.open(args["image"]), P)

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

#Main
if __name__=="__main__":
    #Analisador de argumentos.
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Caminho para a imagem de entrada.")
    ap.add_argument("-model", "--model", type=str, default = "InceptionV3", help = "Nome do modelo a ser usado.")
    args = vars(ap.parse_args())

    #Verifica a validade do modelo passado como argumento.
    if args["model"] not in MODELS.keys():
        raise AssertionError("Argumento --model invalido")

    predict(args)
