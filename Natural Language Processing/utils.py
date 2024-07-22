from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.label_dictionary import LabelDictionary

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import datasets
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


########## Utils: Structured Perceptron ############
####################################################

def prepare_data(df, sentence_col='sentence_id', words_col='words', tags_col='tags'):
    """
    Prepares the data from the given DataFrame, returning lists of tuples.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing sentences and tags.
    - sentence_col (str): The column name for sentence identifiers. Default is 'sentence_id'.
    - words_col (str): The column name for words. Default is 'words'.
    - tags_col (str): The column name for tags. Default is 'tags'.
    
    Returns:
    - X (list of lists): The sentences, each represented as a list of words.
    - y (list of lists): The tags, each represented as a list of tags.
    """
    X = []
    y = []
    
    # Get the unique sentence IDs
    unique_sentence_ids = df[sentence_col].unique()
    
    for sentence_id in unique_sentence_ids:
        # Get the words and tags for the current sentence ID
        words = df[df[sentence_col] == sentence_id][words_col].values.tolist()
        tags = df[df[sentence_col] == sentence_id][tags_col].values.tolist()
        
        # Append to the respective lists
        X.append(words)
        y.append(tags)
    
    return X, y

def create_vocabulary(X, y):
    """
    Generates dictionaries mapping words and tags to unique indices.

    Parameters:
    - X (list of lists): The sentences, each represented as a tuple of words.
    - y (list of lists): The tags, each represented as a tuple of tags.

    Returns:
    - word_pos_dict (dict): A dictionary mapping each unique word to a unique index.
    - tag_pos_dict (dict): A dictionary mapping each unique tag to a unique index.
    """
    # Initialize index counter and dictionaries
    i = 0
    word_pos_dict = {}
    
    # Iterate over all sentences to map each unique word to an index
    for sentence in X:
        for word in sentence:
            if word not in word_pos_dict:
                word_pos_dict[word] = i
                i += 1
                
    # Reset index counter for tags
    i = 0
    tag_pos_dict = {}
    
    # Iterate over all sentences to map each unique tag to an index
    for sentence in y:
        for tag in sentence:
            if tag not in tag_pos_dict:
                tag_pos_dict[tag] = i
                i += 1
                
    tag_pos_dict_rev = {v: k for k, v in tag_pos_dict.items()}  # Reverse tag dictionary

    return word_pos_dict, tag_pos_dict, tag_pos_dict_rev

def create_sequence_list(X, y, word_pos_dict, tag_pos_dict):
    """
    Adds sequences of words and tags to a SequenceList object
    Parameters:
    - X (list of lists): The sentences, each represented as a tuple of words.
    - y (list of lists): The tags, each represented as a tuple of tags.
    - corpus_word_dict (dict): Dictionary mapping words to unique indices.
    - corpus_tag_dict (dict): Dictionary mapping tags to unique indices.

    Returns:
    - seq (SequenceList): The SequenceList object after adding sequences.
    """
    seq =  SequenceList(LabelDictionary(word_pos_dict), LabelDictionary(tag_pos_dict))
    
    for words, tags in zip(X, y):
        seq.add_sequence(words, tags, LabelDictionary(word_pos_dict), LabelDictionary(tag_pos_dict))
    
    return seq


def show_feats(feature_mapper, seq, feature_type=["Initial features", "Transition features", "Final features", "Emission features"]):
    """
    Displays the features of a sequence in a human-readable format, categorizing them into different feature types.

    Parameters:
    - feature_mapper: An object responsible for mapping features to their IDs and providing methods to retrieve features for sequences.
    - seq: The sequence for which the features are to be displayed.
    - feature_type: A list of feature type names. Default is ["Initial features", "Transition features", "Final features", "Emission features"].

    Returns:
      None
    """
    
    inv_feature_dict = {word: pos for pos, word in feature_mapper.feature_dict.items()}

    for feat, feat_ids in enumerate(feature_mapper.get_sequence_features(seq)):
        print(feature_type[feat])

        for id_list in feat_ids:
            for k, id_val in enumerate(id_list):
                print(id_list, inv_feature_dict[id_val])  # Print the feature IDs and their corresponding names

        print("\n")


def get_tiny_test():
    """
    Creates a tiny test dataset.

    Args:
        None

    Returns:
        A tuple containing lists of sentences (TINY_TEST) and tags (TAGS).
    """
    TINY_TEST = [['The programmers from Barcelona might write a sentence without a spell checker . '],
         ['The programmers from Barchelona cannot write a sentence without a spell checker . '],
         ['Jack London went to Parris . '],
         ['Jack London went to Paris . '],
         ['Bill gates and Steve jobs never though Microsoft would become such a big company . '],
         ['Bill Gates and Steve Jobs never though Microsof would become such a big company . '],
         ['The president of U.S.A though they could win the war . '],
         ['The president of the United States of America though they could win the war . '],
         ['The king of Saudi Arabia wanted total control . '],
         ['Robin does not want to go to Saudi Arabia . '],
         ['Apple is a great company . '],
         ['I really love apples and oranges . '],
         ['Alice and Henry went to the Microsoft store to buy a new computer during their trip to New York . ']]

    TAGS = [['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'I-geo', 'I-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'I-geo', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O'],
            ['B-org', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'B-per', 'O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo',
             'I-geo', 'O']]

    return [i[0].split() for i in TINY_TEST], TAGS   


def predict_StructuredPerceptron(model, X):
    """
    Predicts the tags for the input sequences using a StructuredPerceptron model.

    Args:
        model: A trained StructuredPerceptron model.
        X: A list of input sequences (sentences).

    Returns:
        A list of predicted tags for the input sequences.
    """
    y_pred = []

    progress_bar = tqdm(range(len(X)), desc="Predicting tags", unit="sequence")

    for i in progress_bar:
        # Predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(X[i])
        y_pred.append(predicted_tag)

    # Convert each numpy array to a list
    y_pred = [np.ndarray.tolist(array) for array in y_pred]
    
    # Concatenate the lists and flatten
    y_pred = np.concatenate(y_pred).ravel().tolist()

    return y_pred

def accuracy(true, pred):
    """
    Computes the accuracy of predicted tags compared to true tags, excluding instances where true[i] == 'O'.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The accuracy score, which measures the proportion of correct predictions.
    """
    # Exclude 'O' from accuracy prediction
    idx = [i for i, x in enumerate(true) if x != 'O']

    # Get the true and predicted tags for those indexes
    true = [true[i] for i in idx]
    pred = [pred[i] for i in idx]

    return accuracy_score(true, pred)


def plot_confusion_matrix(true, pred, tag_dict_rev):
    """
    Plots a confusion matrix for NER tags.

    Args:
        true: A list or array of true labels.
        pred: A list or array of predicted labels.
        tag_dict_rev: A dictionary mapping label (tags) indices to label (tags) names.

    Returns:
        None
    """
    # Get all unique tag values from true and pred lists
    unique_tags = np.unique(np.concatenate((true, pred)))

    # Create a tick label list with all unique tags
    tick_labels = [tag_dict_rev.get(tag, tag) for tag in unique_tags]

    # Confusion matrix
    cm = confusion_matrix(true, pred)
    
    # Create a mask to exclude 'O' from being highlighted
    mask = np.zeros_like(cm, dtype=bool)
    if 'O' in tag_dict_rev.values():
        o_index = [key for key, value in tag_dict_rev.items() if value == 'O'][0]
        mask[o_index, :] = True
        mask[:, o_index] = True
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def f1_score_weighted(true, pred):
    """
    Computes the weighted F1 score based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The weighted F1 score.
    """
    return f1_score(true, pred, average='weighted')

def results(true, pred, tag_dict_rev):
    """
    Computes and prints evaluation metrics and displays a confusion matrix based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    
    acc = accuracy(true, pred)
    f1 = f1_score_weighted(true, pred)

    print('Accuracy: {:.4f}'.format(acc))
    print('F1 Score: {:.4f}'.format(f1))

    plot_confusion_matrix(true, pred, tag_dict_rev)

def print_tiny_test_prediction(X, model, tag_dict_rev):
    """
    Prints the predicted tags for each input sequence.

    Args:
        X: A list of input sequences.
        model: The trained model used for prediction.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    y_pred = []
    for i in range(len(X)):
        # Predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(X[i])
        y_pred.append(predicted_tag)

    for i in range(len(X)):
        sentence = X[i]
        tag_list = y_pred[i]
        prediction = ''
        for j in range(len(sentence)):
            # Append each word and its corresponding predicted tag to the prediction string
            prediction += sentence[j] + "/" + tag_dict_rev[tag_list[j]] + " "

        print(prediction + "\n")

###############################################################################
###############################################################################

def preprocess_tuples(x):
    return [
        [
            # Reemplaza '.' y ',' en la palabra si es un string, contiene esos caracteres, y su longitud es mayor que 1
            w.replace('.', '').replace(',', '') if isinstance(w, str) and ('.' in w or ',' in w) and len(w) > 1 else w
            for w in inner_list
        ]
        for inner_list in x
    ]

def create_dict (tags):
  index2tag = {}
  tag2index = {}

  index2tag[0] = 'O'
  for i in range (8):
    index2tag [2*i + 1] = tags [i]
    index2tag [2*i + 2] = tags [8+i]


  tag2index = {y: x for x, y in index2tag.items()}

  return index2tag,tag2index

def align_labels_with_tokens(labels, word_ids):
  new_labels = []
  current_word=None
  for word_id in word_ids:
    if word_id != current_word:
      current_word = word_id
      label = -100 if word_id is None else labels[word_id]
      new_labels.append(label)

    elif word_id is None:
      new_labels.append(-100)

    else:
      label = labels[word_id]

      if label%2==1:
        label = label + 1
      new_labels.append(label)

  return new_labels

def tokenize_and_align_labels(x, y, tokenizer):

    new_labels = []
    tokenized_inputs = tokenizer(x, truncation=True, is_split_into_words=True)
    for i, labels in enumerate(y):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs ['labels'] = new_labels

    # print (tokenized_inputs)

    # print (pd.DataFrame(dict (tokenized_inputs)))

    tokenized_inputs = Dataset.from_pandas(pd.DataFrame(dict (tokenized_inputs)))

    return tokenized_inputs, new_labels

def create_labels (y, dic):
    out = []

    for i in range (len(y)):
        aux = []
        for w in y[i]:
            aux.append(dic[w])
        out.append(aux)

    return out

def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def fit_model (model, train_loader, val_loader, num_train_epochs, optimizer, criterion, patience, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop with validation
    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0
        model.train()  # Set the model to training mode
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Assuming the model returns logits

            # Reshape logits and labels to (batch_size * seq_len, num_classes) and (batch_size * seq_len) respectively
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)

            loss = criterion(logits, labels)  # Compute Cross-Entropy Loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_train_epochs}, Training Loss: {avg_loss}")

        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Assuming the model returns logits

                # Reshape logits and labels to (batch_size * seq_len, num_classes) and (batch_size * seq_len) respectively
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)

                loss = criterion(logits, labels)  # Compute Cross-Entropy Loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_train_epochs}, Validation Loss: {avg_val_loss}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
    return model

def reconstruct_sentence(word_tuple):
    sentence = ""
    for word in word_tuple:
        if word in {',', '.', '!', '?', ';', ':'}:  # Add more punctuation if needed
            if sentence and sentence[-1] == ' ':
                sentence = sentence[:-1]  # Remove space before punctuation
            sentence += str(word) + ' '
        else:
            sentence += str(word) + " "
    return sentence.strip()

def reconstruct_sentence_from_tokens(tokens):
    sentence = ""
    for i, token in enumerate(tokens):
        word = token['word']
        if word.startswith("##"):
            sentence += word[2:]  # Append subword without a space
        else:
            if i > 0 and not sentence.endswith(" "):
                sentence += " "  # Add a space before new word if not at start
            sentence += word
    return sentence

def map_tokens_to_original_words(tokens, labels):
    words = []
    word_labels = []
    current_word = ""
    current_label = -100
    for i, token in enumerate(tokens):
        if token['word'].startswith("##"):
            current_word += token['word'][2:]

        else:
            if current_word:
                words.append(current_word)
                word_labels.append(current_label)
            current_word = token['word']
            current_label = labels[i]
    if current_word:
        words.append(current_word)
        word_labels.append(current_label)
    return words, word_labels

def realign_labels_to_words(tokens_with_labels):
    words, labels = map_tokens_to_original_words(
        tokens_with_labels,
        [token['entity'] for token in tokens_with_labels]
    )
    return words, labels

def unir_apostrofe(lista_palabras,label):
    resultado = []
    i = 0
    indices_eliminar = []
    label_out = list(label)
    while i < len(lista_palabras):
        if lista_palabras[i] == "'" and i < len(lista_palabras) - 1: #apostrofe y no es la ultima
            if lista_palabras[i + 1] in ['s','m','ll','re','ve']: #si la siguiente palabra son solo letras y no cosas raras
                nueva_palabra = lista_palabras[i] + lista_palabras[i + 1]
                resultado.append(nueva_palabra)
                i += 1  # Salta la siguiente palabra ya que fue incluida
                indices_eliminar.append(i+1)

            elif lista_palabras[i + 1] == "t" and i > 0 and lista_palabras[i - 1] == "n":
                # Une 'n', el apostrofo y 't' para formar "n't"
                nueva_palabra = lista_palabras[i - 1] + lista_palabras[i] + lista_palabras[i + 1]
                resultado[-1] = nueva_palabra  # Reemplaza 'n' con "n't"
                i += 1  # Salta 't' ya que fue incluida
                indices_eliminar.append(i+1)
                indices_eliminar.append(i-1)


            elif lista_palabras[i + 1]  in ["ite" ,"ites"]: #Shi'ite
                nueva_palabra = lista_palabras[i - 1] + lista_palabras[i] + lista_palabras[i + 1]
                resultado[-1] = nueva_palabra
                i += 1  # Salta 't' ya que fue incluida
                indices_eliminar.append(i+1)
                indices_eliminar.append(i-1)

            else: resultado.append(lista_palabras[i])



        else:
            resultado.append(lista_palabras[i])
        i += 1
    return (resultado, tuple(x for i,x in enumerate(label_out) if i not in indices_eliminar))

def unir_guion(frase,true_frase,label):
  indices = [i for i, x in enumerate(frase) if x in ['-', '~']]
  out = frase
  aux_indice = []
  indices_eliminar = []
  label_out = list(label)


  for indice in indices:
    if(indice == len(frase) -1): continue
    s = frase[indice-1] + frase[indice] + frase[indice+1]

    #print(s)
    if(s in true_frase):
      out[indice] = s
      indices_eliminar.append(indice-1)
      indices_eliminar.append(indice+1)
      label_out[indice] = label_out[indice-1]
    else: continue

  # out = tuple(x for i,x in enumerate(aux) if i not in indices_eliminar)
  # out = list(out)

  # label_out = tuple(x for i,x in enumerate(label_out) if i not in indices_eliminar)
  # label_out = list(label_out)


  # indices_eliminar = []
  for i in indices:
    if(i == len(frase) -1): continue
    if((i+1 in indices) and  (out[i-1] + out[i] + out[i+1] + out[i+2] in true_frase)): #doble guion
      out[i] = out[i-1] + out[i] + out[i+1] + out[i+2]
      indices_eliminar.append(i+1)
      indices_eliminar.append(i-1)
      indices_eliminar.append(i+2)


    if(indice == len(frase) -2): continue
    if( (i+2 in indices) and ( out[i-1] + out[i] + out[i+1] + out[i+2] + out[i+3] in true_frase )): # 46-year-old

      out[i] = out[i-1] + out[i] + out[i+1] + out[i+2] + out[i+3]
      indices_eliminar.append(i-1)
      indices_eliminar.append(i+1)
      indices_eliminar.append(i+2)
      indices_eliminar.append(i+3)

      label_out[i] = label_out[i-1]

  return (tuple(x for i,x in enumerate(out) if i not in indices_eliminar), tuple(x for i,x in enumerate(label_out) if i not in indices_eliminar))

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

def print_transformers_results(relevant_preds, relevant_labels, label_names, tiny=False):

    relevant_preds_flatten = [item for sublist in relevant_preds for item in sublist]
    relevant_labels_flatten = [item for sublist in relevant_labels for item in sublist]

    if tiny:
        labels = ['0', '5', '6', '11', '13', '14']
    else:
        labels = [str(i) for i in range(len(label_names))]


    report = classification_report(relevant_labels_flatten, relevant_preds_flatten, target_names=labels, zero_division = False,output_dict=True)
    # Extract the accuracy and weighted F1-score
    accuracy = report['accuracy']
    f1_weighted = report['weighted avg']['f1-score']

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Weighted F1-Score: {f1_weighted:.2f}")

    conf_matrix = confusion_matrix(relevant_labels_flatten, relevant_preds_flatten, labels= labels)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.8)


    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=label_names, yticklabels=label_names, annot_kws={"size": 'small'})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def process_prediction (ner_results, preprocessed_sentence):
  reconstructed_sentence = reconstruct_sentence_from_tokens(ner_results)
  words, realigned_labels = realign_labels_to_words(ner_results)
  auxpreds = []
  for label in realigned_labels:
    auxpreds.append(str(label[6:]))#guardar solo el numero que nos interesa de la label
  
  apo_sentence, apo_label = unir_apostrofe(words,auxpreds)
  reco_sentence, reco_label =  unir_guion(apo_sentence,preprocessed_sentence,apo_label)

  return reconstructed_sentence, reco_sentence, reco_label