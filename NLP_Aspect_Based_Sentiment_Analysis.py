#!/usr/bin/env python
# coding: utf-8

# # 1. Sentence-Level Aspect Based Analysis

# ## 1.1 Preprocessing
# 
# A1 seeks the truest mapping from sentences to entity-attribute tuples. If neural network N1 takes noisy inputs, the network may fail to learn the true mapping. For example, the sentence ‘we had terrific fun’ is positive whereas ‘they made terrific fun of our appearance’ is negative (albeit archaic). Preprocessing sentences can extract meaningful linguistic information (semantics, syntax, morphology), while de-noising ambiguities and discarding irrelevant information. 
# 
# P1 will parse the XML file F(in) to obtain the laptop review sentences si.
# Then, P1 tokenises (incl. lemmatization), tags, and chunks each sentence. The processed information is
# vectorised (pi) and appended to a file tensor F(out)

# In[ ]:


import xml.etree.ElementTree as ET                                       # Import ElementTree to work with XML files
import nltk                                                              # Import NLTK library
from nltk.tokenize import sent_tokenize, word_tokenize                   # Import sentence and word tokenizers
from nltk.stem import WordNetLemmatizer                                  # Import WordNetLemmatizer for lemmatization
from nltk.corpus import wordnet                                          # Import wordnet for the WordNet lexical database
from nltk import pos_tag, ne_chunk                                       # Import pos_tag for part-of-speech tagging and ne_chunk for named entity chunking
import numpy as np                                                       # Import numpy for array manipulation

nltk.download('maxent_ne_chunker')                                       # Download resources for named entity chunking
nltk.download('punkt')                                                   # Download resources for tokenization
nltk.download('wordnet')                                                 # Download resources for WordNet
nltk.download('averaged_perceptron_tagger')                              # Download resources for part-of-speech tagging
nltk.download('words')                                                   # Download words dataset for natural language processing


# Assume all training and test files are uploaded into the JupyterLab environment (to avoid filepath issues). Open the training file 'Laptops_Test_p1_gold.xml', preprocess the sentences and append the processed sentences to the entity attribute pair in a list.

# In[ ]:


def get_wordnet_pos(tag):                                                # Define a function for mapping from nltk POS tags to wordnet tags
    tag_dict = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ,
        'S': wordnet.ADJ_SAT,
        'I': wordnet.ADJ,
        'T': wordnet.ADJ,
        'D': wordnet.ADV,
        'P': wordnet.ADV
    }
    return tag_dict.get(tag[0], wordnet.NOUN)                           # Return the WordNet POS for the given tag


def process_text(text, sentiment):                                      # Define a function for preprocessing the sentences in the xml file (eventually)
    sentences = sent_tokenize(text)                                     # Tokenize the text into sentences
    lemmatizer = WordNetLemmatizer()                                    # Create an instance of WordNetLemmatizer
    processed_sentences = []

    for sentence in sentences: 
        tokens = word_tokenize(sentence)                                # Tokenize the sentence into words
        pos_tags = pos_tag(tokens)                                      # Assign part-of-speech tags to tokens
        lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags] # Lemmatize tokens
        chunks = ne_chunk(pos_tags)                                     # Perform named entity chunking
        processed_sentences.append({                                    # Append the processed elements to a dict (adjust depending on which produce the most useful training parameters)
            'sentence': sentence,
            #'tokens': tokens,
            #'lemmas': lemmas,
            'pos_tags': pos_tags,                                       
            'chunks': chunks,
            'entity_label': 'None',                                     # Placeholder for entity_label
            'attribute': 'None',                                        # Placeholder for attribute
            'sentiment': sentiment                                      # Add sentiment information
        })

    return processed_sentences                                          # Return the list of processed sentences


# In[ ]:


def main():
    
    """
    Main function to read the XML file, process the reviews, and output the processed sentences.
    :return: Numpy array containing the processed sentences
    """
        
    tree = ET.parse('Laptops_Train_p1.xml')                             # Parse the XML file
    print("XML file parsed:", tree)
    root = tree.getroot()                                               # Get the root element of the XML tree
    print("XML file parsed.")    
    
    output = []                          

    for review in root.iter('Review'):                                  # Iterate through all the 'Review' elements
        print("Processing review.")     
        for sentence_element in review.findall('.//sentence'):          # Find all 'sentence' elements in each review
            text_element = sentence_element.find('text')                
            if text_element is not None:                                # If 'text' element is found
                text = text_element.text                                # Extract the text content
                
                sentiment = 'None'                                      # Initialise sentiment as none

                opinions = sentence_element.findall('.//Opinion')       # Find all opinion sentiments in each sentence
                if opinions:                                            # If opinions are present
                    for opinion in opinions:                            # Iterate through all opinions
                        sentiment = opinion.get('polarity', 'None')     # Get the sentiment polarity if available
                        category = opinion.get('category')              # Get the opinion category if available
                        if category:
                            entity_label, attribute = category.split('#')
                            break
                    
                else:
                    entity_label, attribute = 'None', 'None'
                    
                processed_sentences = process_text(text, sentiment) 
                output.extend(processed_sentences)
                    
                output[-1].update({'entity_label': entity_label, 'attribute': attribute})


    output_list = np.array(output, dtype=object)                        # Convert the output list to a Numpy array
    print(output_list)
    return output_list

output_list = main()                                                    # Execute the main function and store the result in output_list


# ## 1.2 Training and Evaluation
# 
# Given the preprocessed 'output_tensor', we must train a neural network to classify the sentences and their entity#attribute labels to the sentiment categories "positive, negative, and "neutral". This script uses a pretrained transformer "DistilBERT". Training the transformer on the output_tensor will fine-tune its predictions.

# In[37]:


get_ipython().system('pip install torch transformers                                          # import the torch and transformer packages to support the transformer model and its training functions using tensors')
import torch
from torch.utils.data import Dataset, DataLoader                         # import the dataset and dataloader classes for processing model inputs and outputs
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup # import the tokenizer, classification model, weight optimiser and the learning rate schedule

class SentimentDataset(Dataset):                                         # Define a custom dataset class
    def __init__(self, data, tokenizer, max_length):                     # Define the class which supports the data, tokenizer, and max length of tokens passing throught the model
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):                                                   # Method to define the length of the data (no. of dataset samples)
        return len(self.data)

    def __getitem__(self, idx):                                          # Method for obtaining an indexed item from the dataset
        text = self.data[idx]['sentence']                                
        entity_label = self.data[idx]['entity_label']
        attribute = self.data[idx]['attribute']
        input_text = f"{entity_label} {attribute}: {text}"
        
        label_map = {'positive': 0, 'negative': 1, 'neutral': 2, 'None': 3} # Define label mapping from data entity/attributes to sentiments
        label = label_map[self.data[idx].get('sentiment', 'None')]

        encoding = self.tokenizer.encode_plus(                           # Tokenize the input_text
            input_text,                                                  # Takes the input text, adds CLS/SEP to dilineate sequences, truncates the sequence according to maxlength, determines whether to pad the sequence to adhere to a set value (max length), whether to add binary labels (attention mask) to denote padding tokens, and whether to return the encoded sequence as tensors 
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),                # The encoding function returns a dictionary:  flattened version of the input IDs an attention labels and a tensor of the (integer equivalent) of the target label
            'attention_mask': encoding['attention_mask'].flatten(),      
            'label': torch.tensor(label, dtype=torch.long)               
        }


def train(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()                                                # Define a training function with the model, data loader (to give a usable iterator over the dataset), loss function, device (storage location), scheduler (to update the learning rate during training), and n_examples (the training dataset cardinality)
                                                                         # The model.train() function sets the model to training mode (allows batch normalisation, for example)
    correct_predictions = 0                                              # Initialise correct predictions and loss at zero
    total_loss = 0

    for batch in data_loader:                                            # Using the data_loader, iterate over the data batches
        input_ids = batch["input_ids"].to(device)                        # Move the input_ids, attention_mask, and labels to the specified device (defined above)
        attention_mask = batch["attention_mask"].to(device)              
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss                                              # Define 'outputs' as the forward pass function 'model' outputs 
        logits = outputs.logits                                          # Define two variables 'loss' and 'logits' for the loss and logit prediction of each training batch

        total_loss += loss.item()                                        # Cummulatively add the loss of each training batch

        optimizer.zero_grad()                                            # Clear the gradient from the previous iteration
        loss.backward()                                                  # Compute the gradient using backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip the gradient to prevent them exploding/parameters becoming unstable
        optimizer.step()                                                 # Use the optimizer to update the weights
        scheduler.step()                                                 # Update the learning rate according to the scheduler

        _, preds = torch.max(logits, dim=1)                              # Define a correct prediction
        correct_predictions += torch.sum(preds == labels)                # Update the correct predictions with the sum of the predictions that equal the labels 

    return correct_predictions.double() / n_examples, total_loss / len(data_loader) 
                                                                         # Return (the number of correct predictions/dataset cardinality), average loss

def evaluate(model, data_loader, loss_fn, device, n_examples):           # Define the evaluation function similarly to the training function
    model = model.eval()                                                 # Set the model to evaluation mode

    correct_predictions = 0                                              # initialise correct predictions and total loss
    total_loss = 0

    with torch.no_grad():                                                # Disable the gradient calculation for faster computation
        for batch in data_loader:                                        # Iterate over the data loader dataset. 
            input_ids = batch["input_ids"].to(device)                    # Move the input tensor to the device (CPU or GPU)
            attention_mask = batch["attention_mask"].to(device)          # Move the attention mask tensor to the device
            labels = batch["label"].to(device)                           # Move the label tensor to the device

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # Obtain the output from the model
            loss = outputs.loss                                          # Extract the loss from the output
            logits = outputs.logits                                      # Extract the logits from the output

            total_loss += loss.item()                                    # Accumulate the total loss

            _, preds = torch.max(logits, dim=1)                          # Get the predicted class labels
            correct_predictions += torch.sum(preds == labels)            # Count the number of correct predictions

    return correct_predictions.double() / n_examples, total_loss / len(data_loader)



BATCH_SIZE = 16                                                          # Set up training parameters
MAX_LENGTH = 128                                                         # Maximum sequence length for the tokens
EPOCHS = 4                                                               # Number of passes over entire dataset
LEARNING_RATE = 2e-5                                                     
WARMUP_STEPS = 0                                                         # Number of steps to warm up the learning rate scheduler
WEIGHT_DECAY = 0.01                                                      # Regularisation to prevent overfitting
EPSILON = 1e-8                                                           # Term added to the denominator to improve numerical stability when computing Adam optimizer's update step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Define the hosting device for the model

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") # Load pre-trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4).to(device)

train_dataset = SentimentDataset(output_list, tokenizer, MAX_LENGTH)     # Define the training data as the output of SentimentDataset using the preprocessed xml file
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Convert the training dataset to usable PyTorch tensors

optimizer = AdamW(model.parameters(),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=EPSILON) 
total_steps = len(train_loader) * EPOCHS                                 # Set up the optimizer and scheduler. 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
                                                                         # AdamW and get_linear_...warmup are functions taken from the 'transformer' library
    
for epoch in range(EPOCHS):                                              # print the training accuracy and training loss 
    train_acc, train_loss = train(model, train_loader, optimizer, device, scheduler, len(train_dataset))

    print(f"Epoch {epoch + 1}/{EPOCHS} - Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")


torch.save(model.state_dict(), "fine_tuned_model.pt")                    # Save the fine-tuned model


# ## 1.3 Testing 
# 
# The testing function takes the output_tensor with preprocessed text, entity labels, and attributes and passes these elements into the fine-tunec DistilBert model.  

# In[ ]:


def predict_sentiments(model, output_list, tokenizer, device):           # Define a new function to produce predictions for unlabelled preprocessed sentences
    model = model.eval()                                                 # Set the model to evaluation mode
    predictions = []                                                     # Create an empty list for predictions

    for data in output_list:                                             # for the data in the output_list, extract the contents of the output list 
        text = data['sentence']
        entity_label = data['entity_label']
        attribute = data['attribute']
        input_text = f"{entity_label} {attribute}: {text}"               # Concatenate the entity label, attribute and text to form the input text
        
        encoding = tokenizer.encode_plus(                                # Tokenize the input text using the tokenizer (see above)
            input_text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"].to(device)                     # Move the encoded input tensors to the device (CPU or GPU)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():                                            # Disable gradient calculation (see above)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                                      # Extract the logits from the output

        label_map = {0: "positive", 1: "negative", 2: "neutral", 3: "None"}  
        _, preds = torch.max(logits, dim=1)                              # Create a label map to map the predicted labels to their corresponding sentiment classes
        predictions.append(label_map[preds.item()])                      # Append the predicted sentiment to the predictions list

    return predictions


model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4).to(device)
model.load_state_dict(torch.load("fine_tuned_model.pt"))                 # Load the fine-tuned model


sentiment_predictions = predict_sentiments(model, output_list, tokenizer, device)
print(sentiment_predictions)                                             # Predict the sentiment categories!!!!!


# In[ ]:




