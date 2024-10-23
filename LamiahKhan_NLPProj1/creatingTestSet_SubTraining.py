# Credits: https://dev.to/rosejcday/parsing-csv-rows-into-separate-text-files--29lk
#https://pynative.com/python-random-shuffle/
# https://blog.hubspot.com/website/file-handling-python

import random

# Turn labeled docs list to list of tuples
def parseLabels(dataFile):
    with open(dataFile, 'r') as doc:
        # Split each line by space and convert to tuple, for each line in the file
        docTuples = [tuple(line.strip().split(' ')) for line in doc.readlines()]
    return docTuples

def divideLabels(labelTuples, testFraction):
    # Ensure the labels list is shuffled to randomize the split
    random.shuffle(labelTuples)

    # Calculate the split index
    partitionIndex = int(len(labelTuples) * (1 - testFraction))

    # Split the labels into training and test sets
    trainingLabels = labelTuples[:partitionIndex]
    validationLabels = labelTuples[partitionIndex:]
    return trainingLabels, validationLabels

def saveTuplesToFile(tupleList, outputFile):
    with open(outputFile, 'w') as outFile:
        for tpl in tupleList:
            # Join the tuple elements with a space and write to the file followed by a newline
            outFile.write(' '.join(map(str, tpl)) + '\n')

def savePathsToFile(tupleList, outputFile):
    with open(outputFile, 'w') as outFile:
        for tpl in tupleList:
            # Write only the first element of the tuple to the file followed by a newline
            outFile.write(str(tpl[0]) + '\n')

# Ask for input labeled training file
trainingDataFile = input("Enter the name of the file containing labeled list of training documents: ")

# Ask for split ratio
while True:  # Ensure split is a fraction between 0 and 1
    validationFraction = float(input("Enter test size as a decimal value between 0 and 1 (ex: 0.5): "))
    if 0 < validationFraction < 1:
        print(f"Proceeding with a test size of {validationFraction}")
        break
    else:
        print("Invalid input. Please enter a fraction between 0 and 1.")

# Ask for names of output files
trainingSubsetFile = input("Enter the name of the smaller subset of labeled training documents: ")
validationDataFile = input("Enter the name for the labeled validation documents: ")
unlabeledValidationFile = input("Enter the name for the unlabeled validation documents: ")

# Split labeled training set into labeled training and validation sets
subsetLabelsList = parseLabels(trainingDataFile)  # get all rows from labeled training docs file (doc paths, labels)
trainingSubsetList, validationSubsetList = divideLabels(subsetLabelsList, validationFraction)

# Write to output files
saveTuplesToFile(trainingSubsetList, trainingSubsetFile)
saveTuplesToFile(validationSubsetList, validationDataFile)
savePathsToFile(validationSubsetList, unlabeledValidationFile)  # for unlabeled validation docs, only want doc paths
