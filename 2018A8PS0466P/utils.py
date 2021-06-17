import pandas
import numpy
from LiarLiar import arePantsonFire

import seaborn
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

from torch.utils.data import DataLoader
import torch

def create_glove_dict(path_to_text): # 0.75 Marks
    """
    Create the dictionary containing word and corresponding vector. 
    :param path_to_text: Path to Glove embeddings.  
    """
    embeddings = {}
    
    with open(path_to_text,'r', encoding="utf8") as file:
        for line in file:
            val=line.split()
            word=val[0]
            vector=numpy.asarray(val[1:])
            embeddings[word]=vector
    return embeddings

def get_max_length(dataframe, column_number): # 0.75 Marks
    """
    :param dataframe: Pandas Dataframe
    :param column_number: Column number you want to get max value from
    :return: max_length: int
    """
    max_length = 0
    for sent in dataframe.iloc[:,column_number]:
        sent.lower()
        cl=len(word_tokenize(sent))
        if cl > max_length:
            max_length=cl
    return max_length


def visualize_Attenion(attention_matrix):
    """
    Visualizes multihead attention. Expected input shape to [n_heads, query_len, key_len]
    :param attention_matrix:
    :return:
    """
    assert len(attention_matrix.shape) == 3

    for head in range(attention_matrix.shape[0]):
        seaborn.heatmap(attention_matrix[head])
    plt.show()

def infer(model, dataloader):
    """
    Use for inferencing on the trained model. Assumes batch_size is 1.
    :param model: trained model.
    :param dataloader: Test Dataloader
    :return:
    """
    labels = {0: "true", 1: "mostly true", 2: "half true", 3: "barely true" , 4: "false", 5: "pants on fire"}
    model.eval()
    correct = 0
    wrong = 0
    for _, data in enumerate(dataloader):
        statement = data['statement']
        justification = data['justification']
        credit_history = data['credit_history']
        label = data['label']

        prediction = model(statement, justification, credit_history)
        if torch.argmax(prediction).item() == label.item():
            print("Correct Prediction")
            correct+=1
        else:
            print("wrong prediction")
            wrong+=1

        print(labels[torch.argmax(prediction, dim=1).item()])

        print('-------------------------------------------------------------------------------------------------------')
    print(correct/_)
    print(wrong/_)
    
