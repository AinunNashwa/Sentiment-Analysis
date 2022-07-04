  <a><img alt='tf' src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>
 <a><img alt = 'image' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
 <a><img alt = 'image' src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white"></a>
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 
# Sentiment Analysis on Movie Reviews using Deep Learning

### Descriptions
1) The objectives of this project is to develop customer model reviews either it is positive or negative feedbacks
2) Trained with over 60,000 IMDB dataset to categorize positive and negative reviews.
3) Contains 50,000 reviews and 50,000 sentiments
4) Data contains anomalies such as HTML tags, unnecessary characters and combination of uppercase and lowercase alphabets
6) By using RegEx to remove the anamolies: remove HTML tags and unnecessary character and convert all alphabets into lowercase
7) Trained with 10 epochs
8) Methods used: `Word Embedding`,`LSTM`,`Bidirectional`

### Results
`Model`

<img src="static/model.png" alt="model" style="width:300px;height:500px;">

`Classification_Report` & `Confusion_Matrix`

![Screenshot (1314)](https://user-images.githubusercontent.com/106902414/177074980-ebd5eb8d-9a69-4c58-a728-1d3690e00dad.png)


`Training Data` Accuracy versus Loss

![Validation Loss](https://user-images.githubusercontent.com/106902414/177075043-d29718f7-3598-486f-9291-27d8aad9cce0.png)

![Training Loss](https://user-images.githubusercontent.com/106902414/177075062-aec5cc37-737d-4a29-afe1-d0855fbc0453.png)


### Discussion
*Suggestions to Improve the Performance of Model*
1) Apply `Early Stopping` to prevent overfitting
2) Increase `Droput rate` to control overfitting
3) Approach with different architecture such as: `BERT Model`,`Transformer Model`,`GPT3`

### Credits
`You can load the dataset from here`
https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv


