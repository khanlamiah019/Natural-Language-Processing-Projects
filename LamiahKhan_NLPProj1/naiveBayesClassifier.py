# Lamiah Khan, 09/27/2024, NLP Project 1
# Credits: Professor Sable for analyze.pl.
# Skeleton Naive Bayes code adapted from: https://github.com/shanglun/or_posts/blob/master/deeplearn_nlp_wine/naive_bayes.py
# Laplace smoothing code was adapted from: https://github.com/trransom/Laplace-smoothing/blob/master/Laplace_smoothing.py
# note: I edited both github codes severely to fit our context (added Laplace smoothing, input fields, proper reading functionality, etc.)
# Additional resources will be linked below, if used. 

# Importing libraries
import os                   
import string               
import numpy as np          
import nltk                 
from collections import Counter        
from nltk.stem import PorterStemmer      
from nltk.corpus import stopwords       
# If the following nltk packages are NOT downloaded, uncomment lines
# nltk.download('punkt')               
# nltk.download('stopwords')            

def readDocs(rootFolder, docPaths):
    # Reads content of files, converts them to lowercase, and returns them as a dictionary
    # key = filename, value = file content 
    fileContents = {}
    for doc in docPaths:
        # Joins rootFolder with doc to get full path.
        fullFilePath = os.path.join(rootFolder, doc)
        # Opens read mode.
        with open(fullFilePath, 'r') as file:
            fileName = os.path.basename(doc)
            fileContents[fileName] = file.read().lower()
    return fileContents

def extractTest(testFile):
    # Reads file of a list of test file paths and returns a list of file paths.
    testDocs = []
    with open(testFile, 'r') as file:
        for line in file:
            # Strips whitespace and appends each line.
            testDocs.append(line.strip())
    return testDocs

def extractLabels(labelDoc):
    # Reads label file, each line contains file path, and its associated label.
    # Then, returns a dictionary mapping filenames to labels, and a list of file paths.
    docLabel = {}
    docList = []
    with open(labelDoc, 'r') as file:
        for line in file:
            # Splitting each line into file path and label.
            details = line.strip().split()
            path = details[0]
            fileName = os.path.basename(path)
            docLabel[fileName] = details[1]
            docList.append(path)
    return docLabel, docList

def buildVocab(tokenizedFiles):
    # Computes the vocabulary from a dictionary of tokenized files.
    lexicon = set()
    for wordTokens in tokenizedFiles.values():
        # Adds tokens from each file to vocabulary set (duplicates are ignored).
        lexicon.update(wordTokens)
    return lexicon

def tokenizeDoc(docsContent):
    # Tokenizes content of multiple files, filters out stopwords, and stems the words.
    # Returns dictionary where the filename is the key and the tokenized, stemmed words are the values.
    wordTokens = {}
    # I used the Porter Stemmer for word stemming: https://stackoverflow.com/questions/45670532/stemming-words-with-nltk-python
    wordStemmer = PorterStemmer()
    stopwordList = set(stopwords.words('english'))
    for fileName, content in docsContent.items():
        cleanedContent = ''.join([char for char in content if char not in string.punctuation])
        tokens = nltk.word_tokenize(cleanedContent)
        filteredTokens = [wordStemmer.stem(token) for token in tokens if token not in stopwordList]
        wordTokens[fileName] = np.array(filteredTokens)
    return wordTokens

def calcPrevProb(docLabel):
    #Calculates prior probabilities of each label in the dataset, and returns dictionary.
    labFreq = Counter(docLabel.values())
    totalDocs = len(docLabel)
    return {lbl: freq / totalDocs for lbl, freq in labFreq.items()}

# helped choose val for alpha: https://towardsdatascience.com/laplace-smoothing-in-naïve-bayes-algorithm-9c237a8bdece
# Small alpha values: Closer to the original frequency estimates, but still prevent zero probabilities.
# I wanted to avoid over-smoothing, and since 0.055 is quite small, it can be more appropriate for larger datasets where the risk of encountering unseen words is lower
def calcLikelihood(tokenCount, lexicon, docLabel, smoothing= 0.056):
    #Calculates likelihood of each word given a label using Laplace smoothing.Nested dictionary is returned. 
    wordProb = {}
    for label in set(docLabel.values()):
        wordProb[label] = {}
        # Total words with Laplace smoothing.
        totalWord = sum(tokenCount[label].values()) + smoothing * len(lexicon)
        for word in lexicon:
            # Conditional probability: https://teamtreehouse.com/community/getword01
            wordProb[label][word] = (tokenCount[label].get(word, 0) + smoothing) / totalWord
    return wordProb

def categDoc(priorProbs, likelihoods, lexicon, docTokens):
    # Classifies a document based on its tokens using Naive Bayes classification, returns label w/ highest probability.
    # link: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
    sumProbs = {}
    for label in priorProbs.keys():
        logSumProb = np.log(priorProbs[label])
        for word in docTokens:
            if word in lexicon:
                logSumProb += np.log(likelihoods[label].get(word, 1 / (len(lexicon) + 1)))
        # Stores the sum of log probabilities for the label.
        sumProbs[label] = logSumProb
    return max(sumProbs, key=sumProbs.get)

def calcPredict(priorProbs, likelihoods, lexicon, testTokens, testDocs):
    # Classifies test documents and returns a list of predictions with original file paths.
    docsPredict = []
    for i, (fileName, tokens) in enumerate(testTokens.items()):
        predictedLbl = categDoc(priorProbs, likelihoods, lexicon, tokens)
        originalPath = testDocs[i]
        docsPredict.append((originalPath, predictedLbl))
    return docsPredict

def main():
    # Reads input files, tokenizes content, computes probabilities, classifies test documents, and writes the predictions to an output file.
    trainLabelDoc = input("Enter the filename that contains labeled training documents: ")
    testDocList = input("Enter the filename that contains unlabeled test documents: ")

    # Gets the base directory of the training file.
    rootFolder = os.path.dirname(trainLabelDoc)
    # Extracts corpus name from the filename.
    corpus = os.path.basename(trainLabelDoc).split('_')[0]

    # Reads training labels.
    docLabel, trainingDocs = extractLabels(trainLabelDoc)
    # Reads training files.
    trainDocFile = readDocs(rootFolder, trainingDocs)
    # Tokenizes training files.
    tokenizedTrainDocs = tokenizeDoc(trainDocFile)
    # Computes the vocabulary.
    vocab = buildVocab(tokenizedTrainDocs)

    tokenCount = {}
    for label in set(docLabel.values()):
        tokenCount[label] = Counter()
        for docName, tokens in tokenizedTrainDocs.items():
            if docLabel[docName] == label:
                # Organizes the token counts for each label.
                tokenCount[label].update(tokens)

    # Calculates prior probabilities.
    priorProbs = calcPrevProb(docLabel)
    # Calculates word likelihoods for each label.
    wordLikelihoods = calcLikelihood(tokenCount, vocab, docLabel)

    # Reads the list of test file paths.
    testDocs = extractTest(testDocList)
    # Reads test files.
    testDocsContent = readDocs(rootFolder, testDocs)
    # Tokenizes test files.
    tokenizedTestDocs = tokenizeDoc(testDocsContent)

    # Predicts labels.
    predictions = calcPredict(priorProbs, wordLikelihoods, vocab, tokenizedTestDocs, testDocs)
    # Creates output file path.
    outputDoc = os.path.join(rootFolder, f"predicted_{corpus}_test.labels")
    # Writes predictions to the output file.
    with open(outputDoc, 'w') as file:
        for doc in predictions:
            docPath, predictedLbl = doc
            file.write(f"{docPath} {predictedLbl}\n")

if __name__ == '__main__':
    # Entry point of the program.
    main()
