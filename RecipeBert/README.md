# BERT Based Food Recommendation Systems - RecipeBert
 
## Introduction:

<p align="center">
  <img src="image-2.png" alt="Alt text" width="600" height>
</p>

#### Abstract:

While artificial intelligence continues to revolutionize various industries, the smart home industry has not reached the next-generation level yet. For a home to be truly smart, it should fully predict the needs of its residents, responding with appropriate services promptly. Imagine living in a home that not only manages itself but also enhances the daily lives of its residents by offering everyday meals food recommendations based on their taste, culture, weather, diet, their interest in trying new recipes. This vision of living effortlessly, akin to the experience provided by a high-end motel, is what our research aims to achieve. Our proposed system is built upon a BERT-based natural language processing model that predicts recipe similarity. The system ranks the recipes based on the similarity of the words and semantic similarity. Recipes have similar ingredients and procedures are considered similar recipes. The model effectively curates a personalized list of top recipes, taking into account the household's eating patterns over an extended period. It ensures variety by filtering out suggestions that closely mirror recent meals, thereby preventing repetition and keeping the dining experience fresh and exciting. 

<p align="center">
  <img src="image-3.png" alt="Alt text" width="600">
</p>

## Item to Item Similarity model
Important Key Points in Cooking:
- Ingredients : Items used to make the recipe like chicken, tamato, pasta, chocolate, sugar etc 
- Cooking  process: Baking, deep fry, cooking in water, grinding 

<p align="center">
  <img src="image-5.png" alt="Alt text" width="700" >
</p>

#### A simple question for you:
    Why do you think NLP is the way to  go for this problem. Why not traditional models like user to user
    recommendation models ? or feature based recommendation models ? 

- The project is relatively new ( less than 2 years), this solution is perfect to handle limited historical user data case.
- Simple solutions that require minimal feature engineering and allow for the straightforward addition of new items to the system (like simple addotion of openly available new recipe text).

#### Let's try another one:
    OK all great. NLP is the way. But which transformer architecture to choose? Encoder / Decoder/
    Encoder-decoder? 

- Encoders typically perform better on specific tasks rather than general ones. In the context predicting recipes(sentence) similarity a specialized approach using a model like BERT could be quite effective.


### Why Bert?
1] Ingredients similarity - Word similarity: 
    When ingredients are similar they are both high likely to have similar taste. For Example: chicken lovers, and dessert (sugar) lovers.

2] Procedure similarity - Semantic similarity: 
    If the cooking process is similar, they are both likely to have similar texture and taste. Example: Baking foods, No cooking food(Salads), Food/ Half cooked eggs.

## Sentence Bert Architecture Overview/ Sudo Code:

#### Bert:

    #### Algorithm 9: P ← ETransformer(x|θ)
    /* BERT, an encoder-only transformer, forward pass */

    Input: x ∈ V*, a sequence of token IDs.
    Output: P ∈ (0,1)^Nxlength(x), where each column of P is a distribution over the vocabulary.
    Hyperparameters: e_max, L, H, de, dmIp, df ∈ ℕ
    Parameters: θ includes all of the following parameters:
        W_e ∈ ℝ^deXN, W_p ∈ ℝ^deXe_max, the token and positional embedding matrices.
        For l ∈ [L]:
            | W_l, multi-head attention parameters for layer l, see (4),
            | γ_l^1, β_l^1, γ_l^2, β_l^2 ∈ ℝ^de, two sets of layer-norm parameters,
            | W_l^mpl1, b_l^mpl1 ∈ ℝ^dmIpXde, b_l^mpl1 ∈ ℝ^dmIp, W_l^mpl2 ∈ ℝ^deXdmIp, b_l^mpl2 ∈ ℝ^de, MLP parameters.
        W_f ∈ ℝ^dfXde, γ, β ∈ ℝ^df, the final linear projection and layer-norm parameters.
        W_u ∈ ℝ^Nxde, the unembedding matrix.

    1 ℓ ← length(x)
    2 for t ∈ [ℓ] : e_t ← W_e[:,x[t]] + W_p[:,t]
    3 X ← [e1, e2, ... e_ℓ]
    4 for l = 1, 2, ..., L do
    5    X ← X + MHAttention(X|W_l, Mask = 1)
    6    for t ∈ [ℓ] : X[:,t] ← layer_norm(X[:,t]|γ_l^1, β_l^1)
    7    X ← X + W_l^mpl2 GELU(W_l^mpl1 X + b_l^mpl1 1^T) + b_l^mpl2 1^T
    8    for t ∈ [ℓ] : X[:,t] ← layer_norm(X[:,t]|γ_l^2, β_l^2)
    9 end
    10 X ← GELU(W_f X + b_f 1^T)
    11 for t ∈ [ℓ] : X[:,t] ← layer_norm(X[:,t]|γ, β)
    12 return P = softmax(W_u X)

#### Let's try one more:
    Why cannot we use Bert directly for sentence similarity. What is the need of new architecture so 
    a different transformer?
    hint: Think in terms of computation. What is the time complexity for a dataset of size n  

- Finding the two most similar sentences in a dataset of n: This would require us to feed each unique pair through BERT to find its similarity score and then compare it to all other scores. For n sentences would that result in n(n — 1)/2. This turns out to be a real problem if you are trying to integrate this in a real-time environment. A small dataset of only 10,000 sentences would require 49,995,000 passes through BERT, which on a modern GPU would take 60+ hours! This obviously renders BERT useless in most of these scenarios

#### Sentence Bert: 

###### Siamese network:
It is a class of neural network architectures that contain two or more identical subnetworks. "Identical" here means that they have the same configuration with the same parameters and weights. Parameter updating is mirrored across both subnetworks during the training process.

SentenceBERT twin architecture configured for classification.

<p align="center">
  <img src="image-7.png" alt="Alt text" width="300">
</p>


###### Triplet Objective Function:
Given an anchor sentence a, a positive sentence p, and a negative sentence n, triplet loss tunes the network such that the distance between a and p is smaller than the distance between a and n. Mathematically, we minimize the following loss function:

    Loss Function: max(||s_a - s_p|| - ||s_a - s_n|| + ε, 0)

###### Sentence Bert Sudo Code:

        Load Pretrained_BERT_Model
        Prepare Data_Loader with (Anchor_Sentences, Positive_Sentences, Negative_Sentences)

        Compute_Triplet_Loss(Anchor_Embedding, Positive_Embedding, Negative_Embedding, Margin)
            Positive_Distance ← Compute_Euclidean_Distance(Anchor_Embedding, Positive_Embedding)
            Negative_Distance ← Compute_Euclidean_Distance(Anchor_Embedding, Negative_Embedding)
            Triplet_Loss ← Maximum(Positive_Distance - Negative_Distance + Margin, 0)
        Return Triplet_Loss
    
        For each Epoch in Training_Epochs do
            For each (Anchor, Positive, Negative) in Data_Loader do
                Anchor_Embedding ← Get_Sentence_Embedding(Anchor)
                Positive_Embedding ← Get_Sentence_Embedding(Positive)
                Negative_Embedding ← Get_Sentence_Embedding(Negative)

                Triplet_Loss ← Compute_Triplet_Loss(Anchor_Embedding, Positive_Embedding, Negative_Embedding)
                Backpropagate_Error(Triplet_Loss)

                Update BERT_Model_Parameters
            End For

            If Validation_Performance_Improves: Save_Model_Checkpoint          
        


#### Item to Item Similarity Prediction using Bert:

Simple steps:

- Pass sentences, and convert them into embeddings.
- Calculate the distance between sentence embeddings using Euclidean or Cosine similarity.
- We now have a measure of semantic similarity between sentences. 

## Code Demonstration
https://colab.research.google.com/drive/1P1WGhn-EhVT1YC5W74A1fM-jqV0s5J9l

<p align="center">
  <img src="image-6.png" alt="Alt text" width="500">
</p>

## Critical Analysis:
Is the item-to-item recommendation just enough for a smart home application? Recommendations models always work great if we have user-to-user, item-to-item both, and Complementary models(in a few cases).- Writer was not able to integrate other methods because of their users data constraints. But this method can perfectly be implemented in item to item similarity use cases. Example: Doordash showing similar items when the user searched for an item. 
- Writer wants to integrate other recommendation methods in their future work. 

The model is not validated based on real environment with real users recommendation data and it's only validated based on survey's data. Final validation is still pending. 
- Planning to do as a part of next phase of work.

Comparison with chatGPT chain of thoughts. Can this very generalized model provide better score than Bert?  Other try any other encoder-decoder model. 
- Involving chatGPT increases the project maintenance budget. This simple task should not have a high maintenance cost. There is also a scope for data privacy issues. 
Encoder & decoder performance validation?
-   We tried it, but within our resource limitations, we were not able to achieve similar results. Planning to do further evaluation as a part of future work.

Can we boost performance by fine-tuning? 
-   Can be. We are planning for our next phase as we are interested in improving performance. But it can be computationally very costly step to take. 

Validate RobertSentence Transformer Architecture. 
- Yes, as it's one of the other model considered in SentenceTransformer paper, we are interested in validating it's performance. 

## Repo:

https://github.com/DivyaMereddy007/RecipeBert/tree/main/RecipeBert

## Video Recordings: 
https://github.com/DivyaMereddy007/RecipeBert/raw/main/RecipeBert/video1137462603.mp4
https://www.youtube.com/watch?v=9aUtnod9zqE&t=453s


## Resources Links: 
[1]	Item-based Collaborative Filtering with BERT ,Yuyangzi Fu,Tian Wang, https://aclanthology.org/2020.ecnlp-1.8.pdf

[2] https://www.pinecone.io/learn/series/nlp/sentence-embeddings/

[3] https://medium.com/dair-ai/tl-dr-sentencebert-8dec326daf4e

[4] https://www.coursera.org/learn/convolutional-neural-networks/lecture/bjhmj/siamese-network

[4]	Vishnu Nandakumar, Recommendation system using BERT embeddings, https://medium.com/analytics-vidhya/recommendation-system-using-bert-embeddings-1d8de5fc3c56

## Citations:
Paper: MEREDDY DIVYA. (2023). ENABLING NEXT-GENERATION SMART HOMES THROUGH AI-DRIVEN PERSONALIZED FOOD RECOMMENDATIONS. https://doi.org/10.5281/zenodo.8015512 

[1]	Item-based Collaborative Filtering with BERT ,Yuyangzi Fu,Tian Wang, https://aclanthology.org/2020.ecnlp-1.8.pdf

[2]	Nils Reimers, Iryna Gurevych, Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, https://arxiv.org/pdf/1908.10084.pdf

[3]	Nusrat Jahan Prottasha, Abdullah As Sami, Md Kowsher, Saydul Akbar Murad, Anupam Kumar Bairagi, Mehedi Masud, and Mohammed Baz, Transfer Learning for Sentiment Analysis Using BERT Based Supervised Fine-Tuning, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9185586/

[5]	D. Viji, and S. Revathy, A hybrid approach of Weighted Fine-Tuned BERT extraction with deep Siamese Bi – LSTM model for semantic text similarity identification, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8735740/

[6]	Xingyun Xie, Zifeng Ren, Yuming Gu and Chengwen Zhang, Text Recommendation Algorithm Fused with BERT Semantic Information, https://dl.acm.org/doi/abs/10.1145/3507548.3507582

[7]	Itzik Malkiel , Dvir Ginzburg, Oren Barkan, Avi Caciularu, Jonathan Weill, Noam Koenigstein, Interpreting BERT-based Text Similarity via Activation and Saliency Maps, https://dl.acm.org/doi/abs/10.1145/3485447.3512045

[8]	Allyson Ettinger, What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00298/43535/What-BERT-Is-Not-Lessons-from-a-New-Suite-of

[9]	Semantic Similarity with BERT, https://keras.io/examples/nlp/semantic_similarity_with_bert/#inference-on-custom-sentences

[10]	James Briggs, BERT For Measuring Text Similarity, https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1

[11]	PAUL MOONEY , RecipeNLG (cooking recipes dataset, https://www.kaggle.com/code/paultimothymooney/explore-recipe-nlg-dataset

[12]	Kashish Ahuja, Mukul Goel, Sunil Sikka, and Priyanka Makkar, What-To-Taste: A Food Recommendation System, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3670234

[13]	Yasmin Beij, A Literature Review on Food Recommendation Systems to Improve Online Consumer Decision-Making, https://edepot.wur.nl/526224

[14]	Luís Rita, Building a Food Recommendation System, Towards data science, https://towardsdatascience.com/building-a-food-recommendation-system-90788f78691a

[15]	Thi Ngoc Trang Tran, Müslüm Atas, Alexander Felfernig, and Martin Stettinger , Journal of Intelligent Information Systems volume 50, pages 501–526 (2018), An overview of recommender systems in the healthy food domain, https://link.springer.com/article/10.1007/s10844-017-0469-0

[16]	Pratiksha Ashok.Naik, International Journal of Innovative Science and Research Technology, ISSN No:-2456-2165, Volume 5, Issue 8, August – 2020 

[17]	Florian Pecune1*, Lucile Callebert1 and Stacy Marsella1, Designing Persuasive Food Conversational Recommender Systems With Nudging and Socially-Aware Conversational Strategies, https://www.frontiersin.org/articles/10.3389/frobt.2021.733835/full

[18]	RECIPE COURTESY OF INA GARTEN, https://www.foodnetwork.com

[19]	https://arxiv.org/pdf/1904.06690.pdf

[20]	https://paperswithcode.com/dataset/recipenlg#:~:text=The%20benchmarks%20section%20lists%20all,Homepage%20Benchmarks%20Edit

[21]	https://www.researchgate.net/publication/345308878_Cooking_recipes_generator_utilizing_a_deep_learning-based_language_model


