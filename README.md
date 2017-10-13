# Brand detection


The goal of this task is extracting brands from ecommerce sites' titles (such as eBay or Amazon). In this report I intend to share results of experimenting with different machine learning models for this problem. Data used for this tests was collected from eBay database. It has 350k observations which were reduced later after cleaning process. It contains of four columns: title of the product offer, brand included in item specification, information whether the brand is present in the title (every word in the same sequence) and ebay root category for every product offer. Part of this data (30k observations) is included in this repository in train.csv file. This is how it looks like:

![bez tytulu2](https://user-images.githubusercontent.com/32640820/31491341-4e6f855a-af47-11e7-8935-1a11ca318e6c.png)

Data has been cleaned previously. Firstly the observations with more than 4 words in the brand attribute were removed because the majority of them were labeled incorrectly. Brands were also cleaned from punctuations and part of incorrectly labeled brands was also removed manually. After problems with low recall also all of the observations with column present = False, where dropped. Every title was split into tokens and labeled binary ‘0’ or ‘BRAND’. Data after cleaning was reduced to about 160k observations.

Training and testing were being run on two differently split sets of data. First pair of datasets was created by just random sampling (ratio 0.9 to 0.1) and the other split was created in such a way that the test set included titles which brands were only occurring once in the whole dataset, so the system hadn’t encountered them during training process. The second split was like a bigger challenge for the model, because from my experience sometimes models were simply trying to remember exact words considered as brands rather than finding a sequence in the title. So what I intended to do using second split was to avoid overfitting to known brands. As it could be seen in the results there is a major drop in performance for second split for all the models.

In order to find the best machine learning approach for this kind of task 3 models have been created. Let’s analyze how they are created and what input they require.

## Conditional Random Field model
First used model was based on Conditional Random Field. What I am trying to do using this model is splitting the title into single tokens and creating for every of them set of numerical features which are significant in brand recognizing. 
Features used in CRF model which had an impact on improving the model:
-	Token
-	Token position in the sentence
-	Last two characters of the token
-	Is every character upper
-	Is only first character upper
-	Is the token a digit
-	Previous/next token
-	Is previous/next token upper
-	Is previous/next token a title
-	Does a next token contains a digit
-	Is next token a punctuation
-	Is a token at the beginning/end of the sentence
-	Previous/next bigrams
-	Root category for titles (every category is using different vocabulary and have different title structure, f.e. Apple in electronics is not the same as in grocery)

Example of features for one of the words:

![bez tytulu3](https://user-images.githubusercontent.com/32640820/31491344-4f97f8e0-af47-11e7-85c0-c8f258359195.png)

There were also some tests with using other features like including: part of the speech tag for tokens, last characters for previous and next tokens or information is the token an ampersand, but they didn’t provide any significant improvements.
CRF model was fitted using python library called sklearn_crfsuite. Best results for validation data are presented below:

![bez tytulu11](https://user-images.githubusercontent.com/32640820/31491362-587f61f0-af47-11e7-8d24-498beb13deb7.png)

### Evaluation

Results for random split:

![bez tytulu14](https://user-images.githubusercontent.com/32640820/31491368-5ba84cde-af47-11e7-99a8-a3b324230aa5.png)

Results for unknown brands in test data:

![bez tytulu13](https://user-images.githubusercontent.com/32640820/31491365-5a987896-af47-11e7-8607-d6712009d434.png)

In the tables above we can see results for model tested on randomly shuffled data and results for model tested on unknown brands. Essential information for understanding the results is that every word in the brand is treated as one observation. If we would like to know what is the accuracy for detecting whole sequence of tokens in the brand as one observation, the accuracy is reduced to about 0,85 and 0,35 respectively. As we can see the problem of this model is low recall like in most of named-entity recognition projects. It was partly solved by including in datasets only observations with brand in the title (boost from 0.3 to 0.5) but there is still room for improvement.

## 2) NeuroNER model

Second aproach to brand detection problem is using Neural NER model. Model used for this test was cloned from this repository: https://github.com/Franck-Dernoncourt/NeuroNER. NeuroNER was made to find entities in the continuous text but I tried to use it with a few changes in the code to predict brands in not so grammatically structured titles. NeuroNER is based on Bi-directional LSTM networks and contains three layers:
1) Character-enhanced token-embedding layer,
2) Label prediction layer,
3) Label sequence optimization layer.

Model needs specific format in the input. It has to be text file. Every new title should start with the line "-DOCSTART- -X- -X- O". For every token in the title - part of the speech tag, chunk tag and label should be provided in one line of text.

![bez tytulu10](https://user-images.githubusercontent.com/32640820/31491358-57502b8e-af47-11e7-8966-5556a82d63a8.png)

Most important parameters used in training the model:
- Maximum_number_of_epochs = 25
- Dropout_rate = 0.5
- Optimizer = sgd
- Learning_rate = 0.005
- Gradient_clipping_value = 5.0
- Tagging_format = bio
Word embeddings used in the training process were collected from GloVe6B - text file which contains vectors for 6 billions most popular words in Wikipedia and Gigaword. There were also tests with creating new set of vectors trained on ebay titles but they cause decrease in the f1 score.

### Results for testing on randomly shuffled data:

![bez tytulu9](https://user-images.githubusercontent.com/32640820/31491353-55d7836a-af47-11e7-98f3-55c4a55402ca.png)

![bez tytulu6](https://user-images.githubusercontent.com/32640820/31491349-52a1c98a-af47-11e7-989d-7148c275d9e6.png)


### Results for testing on unknown brands:

![bez tytulu5](https://user-images.githubusercontent.com/32640820/31491346-519a1e70-af47-11e7-80f3-b74fee45c047.png)

![bez tytulu8](https://user-images.githubusercontent.com/32640820/31491352-54a8781e-af47-11e7-8f97-f64cf70e54de.png)
 

## 3) BiLSTM model

Last approach was to change NeuroNER model in order to fit better to the problem which was encountered. The conclusion of analyzing the previous model was that maybe it is too complicated and detailed for this particular case. Titles are not so grammatically structured. Most of the words are actually nouns, adjectives or some random numbers, so there is no need to use parts of the speech tags or chunks in our problem.
So in the next model that I’ve created, I only used vector representation of all the tokens as an input. For embeddings again GloVe6B vectors were used. I also somehow combined two previous models by adding CRF layer to the ouput of LSTM network. Results for model tests are gathered in the next section.
Architecture of BiLSTM neural net:

![bez tytulu7](https://user-images.githubusercontent.com/32640820/31491230-e6cd888e-af46-11e7-83ee-d4e8f389525c.png)
 

## Summary of results for every model

Results presented in this section are again the output of two different ways of splitting the data (random split vs unknown brands in the test data). Real value of models' performance is somewhere in the middle of high accuracy for first split and lower accuracy for the second.

![bez tytulu4](https://user-images.githubusercontent.com/32640820/31351411-4428ce3a-ad2b-11e7-9da7-bbe67fb7c7c8.png)
 
The highest scores for both datasets can be observed using LSTM model so the conclusion is that this model should be used for extracting brands in ebay titles.
What CRF is struggle with is low recall. CRF performs properly using first split but it does not generalize well enough in the second split and that's why it is providing unacceptable 40% accuracy for unknown brands.
Results for Neural NER are somewhere in the middle of the other models' but the complexity of the NeuralNER and significantly longer time of training and predicting brands are the reason that it is not recommended to use this model for this particular task.

## Usage

To train the model with new data either with CRF or LSTM model, csv file named 'train.csv' should be included in helper_files folder. In order to start training process there is only a need to run 'training.py' file.

## API
In the repo you can also find an API which enables to query brands from ebay titles. This particular API is based on CRF model. In order to do this user has to provide json file as an input. “Title” and “root_cat” are required as keys in the json file. The output is a list of found brands.

### Requirements:

falcon==1.3.0

uWSGI==2.0.15
