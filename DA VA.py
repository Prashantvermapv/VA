#!/usr/bin/env python
# coding: utf-8

# In[1]:


import srsly


# In[2]:


letters = srsly.read_json('va_letters_anonymized.txt')


# In[22]:


letters[10:20]


# Total = 314,000,
# Containing VA = 82.6% = *259,364 clinical letters of MEH*

# In[4]:


import nltk
from nltk.tokenize import word_tokenize


# In[5]:


text= letters[0]
tokenized_word=word_tokenize(text)
tokenized_word


# In[6]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))


# In[7]:


filtered=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered)


# In[8]:


import string

# punctuations
punctuations=list(string.punctuation)

filtered_tokens=[]

for i in filtered:
    if i not in punctuations:
        filtered_tokens.append(i)
        
print("Filterd Tokens After Removing Punctuations:",filtered_tokens)


# In[9]:


from nltk.probability import FreqDist
fdist = FreqDist(filtered_tokens)
print(fdist)


# In[10]:


import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()


# In[4]:


import regex as re


# In[5]:


re.findall(r'(?<!\d)(?<!\d/)[0-9]/[0-9]{1,2}(?!/?\d)', letters[0])


# In[13]:


VA_total = []
for letter in letters:
    VA = (re.findall(r'(?<!\d)(?<!\d/)[0-9]/[0-9]{1,2}(?!/?\d)', letter))
    for value in VA:
        if value not in VA_total:
            VA_total.append(value)


# In[8]:


VA_all = []
for letter in letters:
    VA = (re.findall(r'((?<!\d)(?<!\d/)[0-9]/[0-9]{1,2}(?!/?\d)|HM|CF|LP|NLP)', letter))
    for value in VA:
        VA_all.append(value)
VA_all


# In[9]:


from collections import Counter
VA = Counter(VA_all)


# In[10]:


VAnew = {x: count for x, count in VA.items() if count >= 8}


# In[11]:


import matplotlib.pyplot as plt
plt.bar(VAnew.keys(), VAnew.values())


# In[13]:


VA.most_common(10)


# # BIO Tagging

# In[80]:


csv2 = []
right = ['right', 'Right', 'RIGHT', 'RE', 'R', 'r','re']
left = ['left', 'Left', 'LEFT', 'LE', 'L', 'l', 'le']
def tag(letter, num):
    x = 'Sentence ' + str(num)
    for word in letter:
        if word in VA_all:
            csv2.append([x, word, 'VA'])
        elif word in right:
            csv2.append([x, word, 'B-LTR'])
        elif word in left:
            csv2.append([x, word, 'B-LTR'])
        else:
            csv2.append([x, word,'O'])
            


# In[62]:


text= letters[0]
word_tokenize(letter)


# In[81]:


text= letters[0]
csv = []
count = 1
for letter in letters[0:50]:
    tag(word_tokenize(letter), count)
    count +=1
#     print(word_tokenize(letter))
#     for word in word_tokenize(letter):
#         csv.append(word)


# In[82]:


import pandas as pd
df = pd.DataFrame(csv2)


# In[58]:


df.to_csv('tag.csv', index = False, header = False)


# In[83]:


df.iloc[20:40]


# In[88]:


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s[1].values.tolist(),
                                                           s[2].values.tolist())]
        self.grouped = self.data.groupby(0).apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[89]:


getter = SentenceGetter(df)


# In[90]:


sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
sentences[0]


# In[92]:


labels = [[s[1] for s in sentence] for sentence in getter.sentences]
print(labels[0])


# In[93]:


tag_values = list(set(df[2].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}


# In[94]:


tag2idx


# In[68]:


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

torch.__version__


# In[69]:


MAX_LEN = 75
bs = 32


# In[70]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)


# In[95]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


# In[96]:


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


# In[97]:


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]


# In[98]:


tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


# In[99]:


input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")


# In[100]:


tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")


# In[101]:


attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]


# In[102]:


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


# In[122]:


tr_inputs = torch.tensor(tr_inputs).to(torch.int64)
val_inputs = torch.tensor(val_inputs).to(torch.int64)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


# In[123]:


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


# In[105]:


import transformers
from transformers import BertForTokenClassification, AdamW

transformers.__version__


# In[106]:


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)


# In[107]:


model.cuda();


# In[108]:


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)


# In[109]:


from transformers import get_linear_schedule_with_warmup

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# In[116]:


from seqeval.metrics import f1_score, accuracy_score


# In[120]:


from tqdm import tqdm, trange


# In[124]:


## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(loss_values, 'b-o', label="training loss")
plt.plot(validation_loss_values, 'r-o', label="validation loss")

# Label the plot.
plt.title("Learning curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# In[ ]:




