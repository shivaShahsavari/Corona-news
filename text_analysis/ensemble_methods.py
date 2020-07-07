import simple_text_classification
import preprocessing
import FastText
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# read the train data 
usa = preprocessing.train_data(23)
usa.to_csv("usa.csv", index=False)
usa['category_id'] = usa['tag'].factorize()[0]

lr_pred = simple_text_classification.cross_validation_logistic_regression(usa)
lr_pred.columns = ['label_id_lr', 'proba_lr', 'label_lr']

ft_pred = FastText.cross_validation_fasttext(usa)
ft_pred.columns = ['label_ft', 'proba_ft','label_id_ft']

# ensemble stacking method 
features = pd.concat([lr_pred['label_id_lr'],ft_pred['label_id_ft']], axis = 1)
labels = usa.category_id
[X_train, X_test, y_train, y_test] = train_test_split(features, labels, test_size=0.30, random_state=0)
model = LogisticRegression(random_state=1)
model.fit(X_train,y_train)
model.score(X_test, y_test)

# pick the class with the highest probability 
predictions = pd.concat([lr_pred[['label_id_lr', 'proba_lr']],ft_pred[['label_id_ft', 'proba_ft']]], axis = 1)
import numpy as np
predictions['final_tag'] = '0'
predictions['maximum'] = predictions[['proba_lr','proba_ft']].idxmax(axis=1)
predictions['final_tag'] = np.where(predictions['maximum']=='proba_lr',predictions['label_id_lr'], predictions['final_tag'])
predictions['final_tag'] = np.where(predictions['maximum']=='proba_ft',predictions['label_id_ft'], predictions['final_tag'])
predictions['real_tag'] = usa.category_id.reset_index(drop = True)
predictions['error'] = np.where(predictions['final_tag'] == predictions['real_tag'], 1, 0)
accuracy = predictions['error'].sum()/len(predictions)
print(accuracy)