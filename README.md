
  <a><img alt='tf' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>
  <a><img alt = 'image' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
  <a><img alt = 'image' src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white"></a>
 
 
# Sentiment Analysis on Movie Reviews using Deep Learning

### Descriptions
1) The objectives of this project is to develop customer model reviews either it is positive or negative feedbacks
2) Trained with over 60,000 IMDB dataset to categorize positive and negative reviews.
3) Contains 50,000 reviews and 50,000 sentiments
4) Trained with 10 epochs
5) Data contains anomalies such as HTML tags, unnecessary characters and combination of uppercase and lowercase alphabets
6) By using RegEx to remove the anamolies: remove HTML tags and unnecessary character and convert all alphabets into lowercase
7) Methods used: `Word Embedding`,`LSTM`,`Bidirectional`

### Results
`Model`

<img src="static/model.png" alt="model" style="width:200px;height:300px;">

`Classification_Report`

`Confusion_Matrix`

`Training Data`

### Discussion
*Suggestions to Improve the Performance of Model*
1) Apply `Early Stopping` to prevent overfitting
2) Increase `Droput rate` to control overfitting
3) Approach with different architecture such as: `BERT Model`,`Transformer Model`,`GPT3`

### Credits
`You can load the dataset from here`
https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv


