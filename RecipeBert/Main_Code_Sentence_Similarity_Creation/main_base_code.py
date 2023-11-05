import sys
import subprocess

# implement pip as a subprocess:
#for packagename in ['bert','pickle','-U sentence-transformers']:
#	subprocess.check_call([sys.executable, '-m', 'pip', 'install', packagename])

#pip3 install bert
#pip3 install torch==1.2.0 torchvision==0.4.0 -f
#pip install -U sentence-transformers

import pandas as pd
import pickle
import numpy as np
data=pd.read_csv('RecipeNLG_dataset.csv')
data['NER']
data.head()

print(data['ingredients'][1:10000])

sentences = data['ingredients']+' '+data['title']+' '+data['directions']+' '+data['source']+' '+data['NER']
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences=sentences[:100]
#model =SentenceTransformer(model_name)
#sentence_vecs=pickle.load(open("sentence_vecs.pkl","rb"))
sentence_vecs=model.encode(sentences)
sentence_vecs.shape
sentence_vecs
#768 hidden stage size
#pickle.dump(sentence_vecs,open("/content/drive/MyDrive/data_colab/sentence_vecs.pkl","wb"))

#sentence_vecs=pd.read_pickle("/content/drive/MyDrive/data_colab/sentence_vecs.pkl")

pickle.dump(sentence_vecs,open("sentence_vecs.pkl","wb"))
print('pickle dumped')
import pickle
import pandas as pd
sentence_vecs=pd.read_pickle("sentence_vecs.pkl")
from sklearn.metrics.pairwise import cosine_similarity


x=cosine_similarity(
    [sentence_vecs[0]],
    sentence_vecs[1:]
)
print(x)

cs=cosine_similarity(sentence_vecs)
recipe_ids=[0,1,2,3]

def return_similar_items(similarities,recipe_ids,topk):
    return np.array(list(sorted(list(zip(recipe_ids,similarities)),reverse=True,key= lambda x:x[1])),
                    dtype=np.dtype({'names':('recipe_ids','similarity'),'formats':('i8','f8')}))

similarities_sorted=np.apply_along_axis(return_similar_items,topk=10,recipe_ids=recipe_ids,arr=cs,axis=1)
similarities_sorted

top_k=10
arr_1=[]
for i in similarities_sorted.tolist():
    arr1=[]
    for ind,v in enumerate(i):
        #if v[0] in current_ids:
        arr1.append(v)
    arr_1.append(arr1)

print('array',arr_1)

arr=list(map(lambda x:x[:top_k],arr_1))
top_k=min(top_k,len(arr[0]))
similarities_sorted_df=pd.DataFrame(arr,columns=range(len(arr[0])))
sim_df=similarities_sorted_df.applymap(lambda x:x[1])
sim_df=sim_df.rename(columns = {i:i+1 for i in range(top_k)})
sim_df.columns=sim_df.columns.astype(str)

sim_df['1'].sort_values(ascending=False)[1:10]
index=sim_df['1'].sort_values(ascending=False)[1:10].index

print(index)









# delte the below

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]

#Encode all sentences
embeddings = model.encode(sentences)

#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)-1):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

print("Top-5 most similar pairs:")
for score, i, j in all_sentence_combinations[0:5]:
    print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))

#https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]
#!pip3 install bert

sentence_vecs

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

#model =SentenceTransformer(model_name)
sentence_vecs=model.encode(sentences)
sentence_vecs.shape
sentence_vecs
#768 hidden stage size

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(
    [sentence_vecs[0]],
    sentence_vecs[1:]

)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

#model =SentenceTransformer(model_name)
sentence_vecs=model.encode(sentences)
sentence_vecs.shape
sentence_vecs
#768 hidden stage size

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(
    [sentence_vecs[0]],
    sentence_vecs[1:]

)



from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence']


#Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)

####### Same code with transformers and torch
model_name='sentence-transformers/bert-base-nli-mean-tockens'

from transformers import AutoTokenizer, AutoModel
import torch
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)

tokens={'input_ids':[],'attention_mask':[]}
for sentence in sentences:
        new_tockens=tokenizer.encode_plus(sentence,max_len=128,truncation=True, padding='max_length',return_tensors='pt')
        new_tockens['input_ids'].append(new_tockens['input_ids'])
        tockens['attention_mask'].append(new_tokens['attention_mask'][0])


outputs=model(**tockens)

####### Same code with transformers and torch
model_name='sentence-transformers/bert-base-nli-mean-tockens'

from transformers import AutoTokenizer, AutoModel
import torch
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)

tokens={'input_ids':[],'attention_mask':[]}
for sentence in sentences:
        new_tockens=tokenizer.encode_plus(sentence,max_len=128,truncation=True, padding='max_length',return_tensors='pt')
        new_tockens['input_ids'].append(new_tockens['input_ids'])
        tockens['attention_mask'].append(new_tokens['attention_mask'][0])


outputs=model(**tockens)

outputs.keys()

embeddings=outpurs.last_hidden_state
embeddings.shape



attention_mask = tokens['attention_mask']
attention_mask.shape


mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
mask.shape

mask

masked_embeddings = embeddings * mask
masked_embeddings.shape

masked_embeddings

summed = torch.sum(masked_embeddings, 1)
summed.shape

summed_mask = torch.clamp(mask.sum(1), min=1e-9)
summed_mask.shape

summed_mask

mean_pooled = summed / summed_mask

mean_pooled









sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = model.encode(sentences)

sentence_embeddings.shape

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)

