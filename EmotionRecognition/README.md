Cosindering the Emotion recognition approach in our project we had the following steps:

1- Using ISEAR dataset (an international survey on Emotion Antecedents and Reactions) as our training dataset
    This dataset consists of 7500 sentences with labed emotions which are in fact 7 basic emotions defined by Plutchik  in "Plutchik wheel of emotions"
    
2- Applying Lemmatization in preprocessing our text News (instead of stemming)
    For your information : Lemmatization is the process of converting a word to its base form. The difference between stemming and lemmatization is, 
    lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often
    leading to incorrect meanings and spelling errors.
    
3- Appying WordNetLemmatizer with appropriate POS tag
    Wordnet is an large, freely and publicly available lexical database for the English language aiming to establish structured semantic relationships between words. 
    It offers lemmatization capabilities as well and is one of the earliest and most commonly used lemmatizers.
    
4- Applying Word2Vec for words vectorization
    Word2Vec is a more recent model that embeds words in a lower-dimensional vector space using a shallow neural network. The result is a set of word-vectors 
    where vectors close together in vector space have similar meanings based on context, and word-vectors distant to each other have differing meanings. 

5- Learning model by Logistic Regression for text classification
    In my experience, I have found Logistic Regression to be very effective on text data and the underlying algorithm is also fairly easy to understand.
    + We tried SVM, Naive Bayes as well, but Logistic Regression had more precision value.

6- Then applying our model on Dutch News which we scraped from 22/04/2020 to 15/05/2020. 
   The result of our emotion recognition is as below:  
     
   <img src="EmotionRecognitionResult.png" width=900 height=400>
