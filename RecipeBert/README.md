# BERT Based Food Recommendation Systems _ RecipeBert
 
## Introduction:

![Alt text](image-2.png)

#### Abstract:
The rapid proliferation of advanced AI technologies has propelled numerous industries forward, but the smart home sector has yet to realize its full potential in the next-generation landscape. A true smart home transcends mere automation, becoming an entity that comprehends and anticipates residents' needs, providing timely and personalized services. This research paper explores the paradigm of a fully intelligent home environment, where residents enjoy a hospitality-like experience while their smart home proactively serves their requirements. One such service can be offering customized food suggestions for daily meals, considering individual preferences, cultural influences, weather conditions, dietary restrictions, and an inclination to explore novel recipes. Our proposed system leverages a state-of-the-art NLP Bert model-based similarity prediction approach to rank recipes based on word and contextual similarities. Recipes that have common ingredients and preparation methods are categorized as similar, while those differing in ingredient types and cooking techniques are classified as not similar. By analyzing 'n' days of historical eating habits, the system generates a curated selection of the top 'k' recipes while avoiding repetition by excluding products consumed within the recent 'm' days (here m<<<<n).

![Alt text](image-3.png)

## Item to Item Similarity model
Important Key Points in Cooking:
    1] Ingredients : Items used to make the recipe like chicken, tamato, pasta, chocolate sugar etc 
    2] Cooking  process: Baking, deep fry, cooking in water, grinding 

![Alt text](image-5.png)

#### A simle question for you:
    Why do you think NLP is the way to  go for this problem. Why not traditional models like user to user recommendation models ? or feature based recommendation models ? 

-
-
#### An other question again:
    OK all great. NLP is the way. But which transformer architecture to choose? Encoder / Decoder / Encoder-decoder? 

### Why Bert?
    1] Ingredients similarity - Word similarity: 
        When ingredients are similar there is a high possibility of liking the food. For Example: chicken lovers, and dessert (sugar) lovers.
    2] Procedure similarity - Context similarity: 
        If the cooking process is similar, there are similarity between the recipes. Example: Baking foods, No cooking food(Salads), Food/ Half cooked eggs.

## Sentence Bert Architecture Overview/ Sudo Code:

#### Sentence bert: 

#### Architecture: 



#### Item to Item Similarity Prediction using Bert:

Simple steps:
    1] Tune your Bert model. Pass sentences, and convert them into embeddings.
    2] Calculate the distance between sentence embeddings using Euclidean or cosine similarity.
    3] We now have a measure of semantic similarity between sentences. 

## Code Demonstraction
https://drive.google.com/file/d/1P1WGhn-EhVT1YC5W74A1fM-jqV0s5J9l/view?usp=sharing

![Alt text](image-6.png)

Divya Notes delete: Compare with formal algorithums

Divya Notes: While bert has showed 0.4 pearson correlation, Sentence bert showed 0.6 correlation with TFIDF. 
With survey based similarity dataset, while bert has showed 0.4 pearson correlation, Sentence bert showed 0.6 correlationwith TFIDF. 

## Critical Analysis:
#### Is the item to item recommendtaion is just enough for a smart home application?
    Recommendations models always work great if we have user to user, item-to-item both, Complementary models(in few cases).
        - Writer was not able to integret other method becuase of their users data constrains. But this method can perfectly be implimented in item to item similarity use cases. Example: Doordash showing similar items when user searched for an item. 
        - Writer wants to integrat other recommendtaions method in their future work. 
    The model is not validated based on real environment with real users recommendtaion data and it's only validated based on survey's data. Final validation is still pending. 
        - Planning to do as a part of next phase of work.
    Comparison with chatGPT chain of thoughts. Can this very generalized model provide better score than bert ?  Other try any other encoder decoder model. 
        - Involving chatGPT increases the project maintance budget. This simple task should not have high maintaince cost. There is also a scopre for data privacy issues. 
        - Encoder & decoder performance validation. Planning to do as a part of future work.
        

## Repo:


## Resources Links:

## Citations:

## Video Recordings: 


## Divya notes:
Expected questions: architecture differences
Parameter count.




