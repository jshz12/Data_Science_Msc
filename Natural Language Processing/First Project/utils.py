import math
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn import *
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from unicodedata import normalize
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.tokenize import word_tokenize
from textstat import flesch_reading_ease, flesch_kincaid_grade
import fasttext
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
nltk.download('punkt')
nltk.download('stopwords')
"""
Simple solution funtions
"""

#Casts each element in the input list to a string.
def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    # assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))
    
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)    
    X_q1q2 = scipy.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################

    return X_q1q2


def get_mistakes(clf, X_q1q2, y):

    ############### Begin exercise ###################
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y 
    incorrect_indices,  = np.where(incorrect_predictions)
    
    ############### End exercise ###################
    
    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions
    


"""
-----------------------------------------------------------------------------------------------
----------------------------------------Text Cleaning------------------------------------------
-----------------------------------------------------------------------------------------------
"""

contraction_mapping = {
    "a.k.a.": "also known as",
    "abt": "about",
    "acct": "account",
    "adios": "goodbye",
    "afaik": "as far as I know",
    "afk": "away from keyboard",
    "alot": "a lot",
    "ama": "ask me anything",
    "asap": "as soon as possible",
    "atm": "at the moment",
    "b/c": "because",
    "b4": "before",
    "bbl": "be back later",
    "bbs": "be back soon",
    "bff": "best friends forever",
    "bk": "back",
    "brb": "be right back",
    "btw": "by the way",
    "cya": "see you",
    "diy": "do it yourself",
    "dm": "direct message",
    "dnd": "do not disturb",
    "e.g.": "for example",
    "etc": "et cetera",
    "fomo": "fear of missing out",
    "ftw": "for the win",
    "fyi": "for your information",
    "gtg": "got to go",
    "hmu": "hit me up",
    "hbu": "how about you",
    "idk": "I don't know",
    "irl": "in real life",
    "jk": "just kidding",
    "l8r": "later",
    "lol": "laugh out loud",
    "lmk": "let me know",
    "nvm": "never mind",
    "omg": "oh my god",
    "omw": "on my way",
    "otp": "on the phone",
    "pls": "please",
    "ppl": "people",
    "rly": "really",
    "rn": "right now",
    "smh": "shaking my head",
    "sry": "sorry",
    "tbh": "to be honest",
    "tldr": "too long; didn't read",
    "ttyl": "talk to you later",
    "w/": "with",
    "w/o": "without",
    "wbu": "what about you",
    "wfh": "work from home",
    "wym": "what do you mean",
    "yolo": "you only live once",
    "yw": "you're welcome",
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def expand_contractions(text, contraction_mapping=contraction_mapping):
    """
    Expand contractions and abbreviations in text.
    contraction_mapping (dict): Dictionary mapping contractions to their expanded forms.
    """
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

def remove_punctuation(text):
    """
    Remove punctuation from text.
    """
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

def spell_check(text):
    """
    Correct spelling mistakes in text using the pyspellchecker library.

    """
    # Create a SpellChecker object for English
    spell_checker = SpellChecker(language='en')
    
    # Perform spell checking
    corrected_text = []
    for word in text.split():
        # Get the correction for the word
        corrected_word = spell_checker.correction(word)
        if corrected_word is not None:
            # If correction is not None, use it
            corrected_text.append(corrected_word)
        else:
            # If correction is None, keep the original word
            corrected_text.append(word)

    # Join the corrected words back into a single string
    corrected_text = ' '.join(corrected_text)
    
    return corrected_text

def remove_stopwords(text, language='english'):
    """
    Remove stopwords from text.
    """
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def remove_accents(text):
    """
    Remove accents from characters in text.
    """
    return unidecode.unidecode(text)


def normalize_spaces(text):
    """
    Normalize spaces in text.
    """
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text


def text_cleaning(text):
    """
    Applies all the above functions
    """
     # Expand contractions
    text = expand_contractions(text)
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Spell check
    #text = spell_check(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    # Remove accents
    text = remove_accents(text)
    
    # Normalize spaces
    text = normalize_spaces(text)
    
    return text



class BKTreeNode:
    def __init__(self, word):
        self.word = word
        self.children = {}

def levenshtein_distance(a, b):
    if not a: return len(b)
    if not b: return len(a)
    return min(levenshtein_distance(a[1:], b[1:])+(a[0] != b[0]), levenshtein_distance(a[1:], b)+1, levenshtein_distance(a, b[1:])+1)

class BKTree:
    def __init__(self, distance_function):
        self.root = None
        self.distance = distance_function

    def add(self, word):
        if self.root is None:
            self.root = BKTreeNode(word)
            return

        node = self.root
        while True:
            dist = self.distance(word, node.word)
            if dist in node.children:
                node = node.children[dist]
            else:
                node.children[dist] = BKTreeNode(word)
                break

    def search(self, query, max_distance):
        results = []

        def search_node(node, distance):
            dist = self.distance(query, node.word)
            if dist <= max_distance:
                results.append((node.word, dist))

            for d in range(dist - max_distance, dist + max_distance + 1):
                child = node.children.get(d)
                if child is not None:
                    search_node(child, max_distance)

        if self.root is not None:
            search_node(self.root, max_distance)

        return results

class SpellChecker:
    def __init__(self, distance_function=levenshtein_distance):
        self.bktree = BKTree(distance_function)

    def add_words(self, words):
        for word in words:
            self.bktree.add(word)

    def spellcheck(self, word, max_distance):
        return self.bktree.search(word, max_distance)

    def spellcheck_dataframe(self, dataframe, column_name, max_distance):
        misspelled_words = []
        for _, row in dataframe.iterrows():
            for word in row[column_name].split():
                corrections = self.spellcheck(word, max_distance)
                if not corrections:
                    misspelled_words.append((word, row.name))  # Append the misspelled word and its index
        return misspelled_words



def print_results(model,X,y):
    y_proba=model.predict_proba(X)[:, 1]

    find_threshold(y_proba, y, plot_roc=True)

    y_pred=model.predict(X)

    cm = confusion_matrix(y,y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])

    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    print (classification_report(y,y_pred))

    accuracy = accuracy_score(y,y_pred)
    f1 = f1_score(y,y_pred, average='weighted')
    precision = precision_score(y,y_pred, average='weighted')
    recall = recall_score(y,y_pred, average='weighted')

    results = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
    return results






"""
-----------------------------------------------------------------------------------------------
----------------------------------------Feature Extraction------------------------------------------
-----------------------------------------------------------------------------------------------

"""

def updatequestions(questions_df, X,y):
    X = X.merge(questions_df, left_on='qid1', right_on='qid', how='left').drop(columns=['qid'])
    X = X.merge(questions_df, left_on='qid2', right_on='qid', how='left', suffixes=['_1', '_2']).drop(columns=['qid'])

    # replace the values in the question1 and question2 columns
    X['question1'] = X['question_1']
    X['question2'] = X['question_2']

    # drop the additional question_1 and question_2 columns
    X = X.drop(columns=['question_1', 'question_2'])

    # remove empty questions
    mask = (X['question1'].str.len() > 0) & (X['question2'].str.len() > 0)

    X = X[mask]
    y = y[mask]
    X.reset_index(drop=True, inplace=True)
    return X,y

def updateembeddings(questions_df, X):
    X = X.merge(questions_df, left_on='qid1', right_on='qid', how='left').drop(columns=['qid'])
    X = X.merge(questions_df, left_on='qid2', right_on='qid', how='left', suffixes=['_1', '_2']).drop(columns=['qid'])
    # drop the question1 and question2 columns
    X = X.drop(columns=['question1', 'question2'])
    distances = X.apply(lambda row: distance_embeddings(row['embedding_1'], row['embedding_2'], row.name), axis=1)
    # Extract the individual distances from the resulting tuples
    X['cosine_distance'], X['euclidean_distance'], X['manhattan_distance'] = zip(*distances)
    # Remove the columns from the training set
    X = X.drop(columns=['embedding_1', 'embedding_2'])
    return X

# Function to extract features
def extract_features(question1, question2):
    # Extract first words of each question
    first_word_q1 = question1.split()[0].lower()
    first_word_q2 = question2.split()[0].lower()

    # Check if first words are equal
    first_word_equal = int(first_word_q1 == first_word_q2)

    # Tokenize questions
    tokens_q1 = word_tokenize(question1.lower())
    tokens_q2 = word_tokenize(question2.lower())

    # Calculate common words ratio
    common_words_ratio = len(set(tokens_q1) & set(tokens_q2)) / max(len(set(tokens_q1)), len(set(tokens_q2)))

    # Compute Flesch reading ease score and Flesch–Kincaid grade level
    flesch_score_q1 = flesch_reading_ease(question1)
    flesch_score_q2 = flesch_reading_ease(question2)
    flesch_grade_q1 = flesch_kincaid_grade(question1)
    flesch_grade_q2 = flesch_kincaid_grade(question2)

    return {
        'first_word_equal': first_word_equal,
        'common_words_ratio': common_words_ratio,
        'flesch_reading_ease_q1': flesch_score_q1,
        'flesch_reading_ease_q2': flesch_score_q2,
        'flesch_kincaid_grade_q1': flesch_grade_q1,
        'flesch_kincaid_grade_q2': flesch_grade_q2
    }

"""
-----------------------------------------------------------------------------------------------
----------------------------------------Word2vec and Fasttext Embeddings----------------------------------
-----------------------------------------------------------------------------------------------

"""

import numpy as np
import pandas as pd
import gensim.downloader as api  #word2vec
import fasttext.util
from scipy.spatial.distance import cosine,euclidean,cityblock #distances

"""
---------------------------------------------------------------------------------------------------------------------------

"""

def word2vec_embeddings(text,model):

    # Lowercase and split it into individual words
    words = text.lower().split()

    # Get embeddings for all words that are present in the model's vocabulary
    embeddings = [model.get_vector(word) for word in words if word in model.index_to_key]

    # If no embeddings were found, return a null embedding
    if len(embeddings) == 0:
        return None
    else:
        # Take the mean of all embeddings to get a single embedding for the entire text
        return np.mean(embeddings, axis=0)


"""
---------------------------------------------------------------------------------------------------------------------------

"""

def fasttext_embeddings(text,model):

    # Lowercase and split it into individual words
    words = text.lower().split()

    # Get embeddings for all words that are present in the model's vocabulary
    embeddings = [model.get_word_vector(word) for word in words if word in model]
    # If no embeddings were found, return None
    if len(embeddings) == 0:
        return None
    else:
        # Take the mean of all embeddings to get a single embedding for the entire text
        return np.mean(embeddings, axis=0)

"""
---------------------------------------------------------------------------------------------------------------------------

"""


#Cosine Distance

def cos_distance(embedding1, embedding2):

    if embedding1 is None or embedding2 is None:
        return None
    else:
        # Compute the cosine distance between the two embeddings
        return cosine(embedding1, embedding2) #we want the distance, if we set 1 - cosine(embedding1, embedding2), for two identical questions we will obtain 1 as an output

#Euclidean Distance

def euclidean_distance(embedding1, embedding2):

    if embedding1 is None or embedding2 is None:
        return None
    else:
        # Compute the Euclidean distance between the two embeddings
        return euclidean(embedding1,embedding2)

def manhattan_distance(embedding1, embedding2):

    if embedding1 is None or embedding2 is None:
        return None
    else:
        # Compute the Manhattan distance between the two embeddings
        return cityblock(embedding1,embedding2)


"""
---------------------------------------------------------------------------------------------------------------------------

"""



def distance_embeddings(embedding1, embedding2, index):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    if embedding1.shape != (100,) or embedding2.shape != (100,):
        print("Row:", index)
        print("embedding1 shape:", embedding1.shape)
        print("embedding2 shape:", embedding2.shape)

    # Compute distances between corresponding elements in the embeddings
    distances_cos = cosine(embedding1, embedding2)
    distances_euc = euclidean(embedding1, embedding2)
    distances_manh = cityblock(embedding1, embedding2)

    return distances_cos, distances_euc, distances_manh

"""
---------------------------------------------------------------------------------------------------------------------------

"""

def training_format_sup(x,y,char):
  with open(char, 'w') as f:
    for i, question_pair in enumerate(x):
        label_value = y[i]
        f.write("__label__{} {} {}\n".format(label_value, question_pair[0], question_pair[1]))


"""
---------------------------------------------------------------------------------------------------------------------------

"""



def questions_fasttext_format(questions):
  combined_questions = []

  for pregunta_pair in questions:
    aux = pregunta_pair[0] + ' ' + pregunta_pair[1]
    combined_questions.append(aux.replace('\n', ''))

  return(combined_questions)



"""
---------------------------------------------------------------------------------------------------------------------------

"""


def predict_fasttext(questions,loaded_model):
  predicted_label = []
  predicted_prob= []
  for i in range(len(questions)):
    aux_label, aux_prob = loaded_model.predict(questions[i])
    predicted_label.append(int(aux_label[0][-1]))
    if (predicted_label[i] == 0):
      predicted_prob.append(1-float(aux_prob[0]))
    else:
      predicted_prob.append(float(aux_prob[0]))

  return(predicted_label, predicted_prob)


"""
---------------------------------------------------------------------------------------------------------------------------

"""


def format_data(questions, output_file):
    with open(output_file, 'w') as f:
        for question in questions:
            f.write("{}\n".format(question))

"""
---------------------------------------------------------------------------------------------------------------------------

"""


def get_results(preds,probs, label): 
  _ = find_threshold(probs,label)
  cm = confusion_matrix(label,preds)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])

  disp.plot(cmap=plt.cm.Blues)
  plt.title('Confusion Matrix')
  plt.show()

  print (classification_report(label, preds))
  accuracy = accuracy_score(label,preds)
  f1 = f1_score(label,preds, average='weighted')
  precision = precision_score(label,preds, average='weighted')
  recall = recall_score(label,preds, average='weighted')

  results = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

  return(results)










"""
-----------------------------------------------------------------------------------------------
----------------------------------------Sentence Transformers----------------------------------
-----------------------------------------------------------------------------------------------

"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn, optim
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, classification_report
def fit (model, df_train, df_val, loss='ContrastiveLoss', out_model=None, margin = 0.5, batch_size = 128, epochs = 8):
  train_examples = [InputExample(texts=[df_train["question1"][i], df_train["question2"][i]], label=float (df_train["is_duplicate"][i])) for i in df_train.index]
  val_examples = [InputExample(texts=[df_val['question1'][i], df_val['question2'][i]], label=float(df_val['is_duplicate'][i])) for i in df_val.index]

  train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=128, num_workers=2, pin_memory=True)

  distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

  if loss == 'OnlineContrastiveLoss':
    train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

  else:
    train_loss = losses.ContrastiveLoss(model=model,margin=margin)
  '''
  Contrastive loss Expects as input two texts and a label of either 0 or 1.
  If the label == 1, then the distance between the two embeddings is reduced.
  If the label == 0, then the distance between the embeddings is increased.
  Uses siamese distance metric (1- cosine).
  '''
  evaluator = BinaryClassificationEvaluator.from_input_examples(val_examples, show_progress_bar = True, batch_size=batch_size)

  model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            evaluator=evaluator,
            evaluation_steps=500,
            save_best_model = True,
            output_path = os.path.join(home_dir, out_model))

def find_threshold (preds, label, plot_roc=True):
  '''
  To find the optimal threshold using roc curve, so that the euclidean distance to the operating point is minimal
  '''
  # Compute ROC curve
  fpr, tpr, thresholds = roc_curve(label, preds)

  # Vectorized calculation of Euclidean distance from perfect classifier point (0,1)
  distance_perfection = np.sqrt(fpr**2 + (1 - tpr)**2)

  # Find index of minimum distance
  min_index = np.argmin(distance_perfection)

  # Select corresponding threshold
  threshold = thresholds[min_index]

  print ('Threshold: ', threshold)

  if plot_roc:
    print ('AUC: %lf' % (auc(fpr, tpr)))
    plt.figure()
    plt.scatter(fpr, tpr, c= thresholds, cmap='viridis', vmin=0, vmax=1)
    clb = plt.colorbar()
    clb.ax.set_title('Threshold')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label= 'Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

  return threshold

def predict (model, df, debug=False, threshold=None, show_roc=False, print_cm=True):

  '''
  Prediction of a dataframe using a threshold. If not provided, we determine it with the function above. Returns the threshold
  '''

  sentence1 = [x for x in df["question1"]]
  sentence2 = [x for x in df["question2"]]

  if debug:
    print ('Encoding sentence1')

  sentence1_embeddings = model.encode(sentence1)

  if debug:
    print ('Encoding sentence2')
  sentence2_embeddings = model.encode(sentence2)

  if debug:
    print ('Calculating distances')

  dist = [cosine_similarity(sentence1_embeddings[i].reshape(1,-1), sentence2_embeddings[i].reshape(1,-1))[0][0] for i in range (len(df))]

  preds = (dist - min(dist))/(max(dist) - min(dist))

  if threshold == None:
    threshold = find_threshold (preds, df['is_duplicate'], plot_roc=show_roc)

  predictions = [0 if x <= threshold else 1 for x in preds]

  if print_cm:

    cm = confusion_matrix(df['is_duplicate'],predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])

    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    print (classification_report(df['is_duplicate'], predictions))

  accuracy = accuracy_score(df['is_duplicate'],predictions)
  f1 = f1_score(df['is_duplicate'],predictions, average='weighted')
  precision = precision_score(df['is_duplicate'],predictions, average='weighted')
  recall = recall_score(df['is_duplicate'],predictions, average='weighted')

  results = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

  return predictions, threshold, results

"""
-----------------------------------------------------------------------------------------------
----------------------------------------TFIDF----------------------------------
-----------------------------------------------------------------------------------------------

"""
import math
import numpy as np
from scipy.sparse import csr_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


class TFIDF:
    def __init__(self):
        self.corpus = None
        self.num_documents = None
        self.vocab = None
        self.term_idf_cache = {}
        self.tfidf_scores = None
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def calculate_tf(self, term, document):
        word_count = document.count(term)
        total_words = len(self.split(document))
        return word_count / total_words if total_words > 0 else 0

    def calculate_idf(self, term):
        if term in self.term_idf_cache:
            return self.term_idf_cache[term]

        num_documents_with_term = self.doc_freq.get(term, 0)
        idf = math.log(self.num_documents / (1 + num_documents_with_term))
        self.term_idf_cache[term] = idf
        return idf

    def fit(self, corpus):
        self.corpus = corpus
        self.num_documents = len(corpus)
        self.vocab = set()  # Initialize vocabulary set
        self.doc_freq = {}  # Dictionary to store document frequency for each term \ number of documents containing term t
        
        # Construct vocabulary set and compute document frequency \ number of documents containing term t
        for document in self.corpus:
            terms = self.split(document)
            self.vocab.update(terms)
            for term in set(terms):  # Use set to count each document only once
                if term not in self.stopwords: # Exclude stopwords from vocabulary
                    self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        # Precompute term indices
        self.term_indices = {term: idx for idx, term in enumerate(self.vocab)}
            
    def transform(self, questions):
        tfidf_data = []
        tfidf_row = []
        tfidf_col = []

        # Iterate over documents and terms to compute TF-IDF scores
        for i, document in enumerate(questions):
            for term in set(self.split(document)):
                if term in self.vocab:  # Check if term exists in vocabulary, if not we will just work with tokens from seen during fit
                    if term not in self.term_idf_cache:
                        self.term_idf_cache[term] = self.calculate_idf(term)

                    tfidf_score = self.calculate_tf(term, document) * self.term_idf_cache[term]
                    tfidf_data.append(tfidf_score)
                    tfidf_row.append(i)
                    tfidf_col.append(self.term_indices[term])

        self.tfidf_scores = csr_matrix((tfidf_data, (tfidf_row, tfidf_col)), shape=(len(questions), len(self.vocab)))
      
        return self.tfidf_scores
    
    def split(self, text, token_pattern="\\b\\w\\w+\\b"):
        # Split text into terms using the provided token pattern
        terms = re.findall(token_pattern, text.lower())
        lemmatized_terms = [self.lemmatizer.lemmatize(term) for term in terms]
        return [term for term in lemmatized_terms if term not in self.stopwords]  # Exclude stopwords after lemmatization

def perform_grid_search(X_train, y_train, X_val, y_val, param_grid):
    # Initialize Logistic Regression model
    log_reg = sklearn.linear_model.LogisticRegression(random_state=123, max_iter=100000)

    # Perform grid search
    best_score = 0
    best_params = None
    for C in param_grid['C']:
        for solver in param_grid['solver']:
            for penalty in param_grid['penalty']:
                log_reg.set_params(C=C, solver=solver, penalty=penalty)
                log_reg.fit(X_train, y_train)
                score = log_reg.score(X_val, y_val)

                # Check if this is the best score
                if score > best_score:
                    best_score = score
                    best_params = {'C': C, 'solver': solver, 'penalty': penalty}

    return best_score, best_params



def plot_roc_curve_and_threshold(tfidf_data, y_true, clf, optimal_threshold=None, calculate_roc=True, calculate_optimal_threshold=False):

    results = {}
    # Extract probabilities for the positive class
    y_score = clf.predict_proba(tfidf_data)[:, 1]

    roc_auc_score = None
    accuracy = None
    precision = None
    recall = None
    f1_score = None

    # Calculate ROC AUC score if requested
    if calculate_roc:
        roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_score)
        print("ROC AUC score:", roc_auc_score)

        # Generate ROC curve if requested
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
        thresholds_clipped = np.clip(thresholds, 0, 1)

        # Plot ROC curve with thresholds as gradient color
        plt.figure()
        plt.scatter(fpr, tpr, c=thresholds_clipped, cmap='viridis', s=5)
        plt.colorbar(label='Threshold')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve with Thresholds')
        plt.legend(loc="lower right")
        plt.show()

    # Print the optimal threshold if provided
    if calculate_optimal_threshold:
        if optimal_threshold is None:
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds_clipped[optimal_idx]
            print("Optimal Threshold:", optimal_threshold)

    # Apply custom threshold to obtain binary predictions and calculate classification report
    if optimal_threshold is not None:
        y_pred_custom = (y_score >= optimal_threshold).astype(int)

        # Calculate classification report
        print(sklearn.metrics.classification_report(y_true, y_pred_custom))
        report = sklearn.metrics.classification_report(y_true, y_pred_custom, output_dict=True)

        # Extract relevant metrics from the classification report
        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']

        results = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1_score}

    # Return the metrics and optimal threshold
    return optimal_threshold, results