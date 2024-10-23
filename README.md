# ECE-467
Projects in Natural Language Processing

Project 1: 
Lamiah Khan 10/10/2024 NLP: Text Categorization Project
I. System Functionality
The text categorization program was developed using Anaconda’s Spyder, with programming language Python 3.11. As mentioned in naiveBayesClassifier.py, libraries needed for this program include OS(standard library),String(standard library),numpy(pip install numpy), collections(standard library), and NLTK, along with NLTK packages "punkt" and "stopwords":
# nltk.download('punkt')
# nltk.download('stopwords') In order to run the program:
1. Download the given TC_provided folder into an available directory.
2. Run naiveBayesClassifier.py.
3. 1st input example: “Enter the filename that contains labeled training documents:
corpus1_train.labels.”
4. 2nd input example: “Enter the filename that contains unlabeled test documents: corpus1_test.list.”
5. Run analyze.pl to compare predicted model’s accuracy: “!perl analyze.pl
predicted_corpus1_test.labels corpus1_test.labels”
II. ML Method: Naive Bayes
Naive Bayes was utilized to implement text categorization. The system tokenized training and test files through the tokenizeDoc function, which included converting document’s content to lowercase, and removing punctuation. This reduced the vocabulary size, and improved the generalization of the system. Next, the function applies tokenization using nltk.word_tokenize. Afterwards, the algorithm filtered stopwords, or common words such as “the” and “is,” using the nltk.corpus.stopwords list. This greatly improved the model’s accuracy by excluding non-informative words, and hence generating word probabilities better. Lastly, the function incorporated stemming using NLTK’s PorterStemmer. This significantly improved the model's accuracy due to the improvement in generalizing different word forms. Specifically, its inclusion was useful for corpus 2&3, which were larger datasets.
The system implements Laplace smoothing in the calcLikelihood function, with alpha = 0.056, which was determined from multiple tests. The value of alpha is small to prevent excessive smoothing, which tends to occur when alpha = 1.
Lastly, the system’s performance for corpus 2 & 3 are evaluated with the creatingTestSet_SubTraining.py algorithm, which utilizes split ratios to divide the provided training sets into “sub-training sets,” and a “test set.” The user was prompted to enter a split ratio between 0 and 1. For the Results, approximately 1⁄3 of the training test was utilized for testing, while 2⁄3 was used as the sub-training set.
<img width="454" alt="Screenshot 2024-10-23 at 7 25 03 PM" src="https://github.com/user-attachments/assets/9e3fc557-cbd7-4e6d-b2e3-a88358fc4254">

