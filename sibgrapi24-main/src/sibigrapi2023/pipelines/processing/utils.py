import enchant
import re

def cleanText(t):
    t = re.sub(r" +", " ", t) # replace multiple spaces with one
    t = re.sub(r"[\s\n]+\n", "\n", t).strip() # remove empty lines
    return t

def accuracyByLevenshteinDistance(textOCR,label):
    #tira espaços, tab e \n
    text = cleanText(textOCR)
    labelText = cleanText(label)

    #calcula a distancia de levenshtein entre os textos
    levDistance = enchant.utils.levenshtein(text, labelText)

    #calcula a acurácia fazendo accuracy = n-erros/n
    accuracy = (len(labelText)-levDistance) / len(labelText)
    return accuracy
