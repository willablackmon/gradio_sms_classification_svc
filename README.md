## Gradio SMS Classification Function, Powered by a Support Vector Classification (SVC) Model

# SMS Spam Classification with Gradio Interface

This project implements an SMS spam classification system using machine learning techniques. It leverages a **Support Vector Machine (SVM)** classifier and **TF-IDF Vectorization** to differentiate between spam and non-spam (ham) messages. The classification model is deployed using **Gradio** for a simple and interactive web interface.

## Goals

The objective of this project is to:

* Build a machine learning model to classify SMS messages as either spam or not spam.
* Create an intuitive web interface using Gradio for real-time predictions.
* Provide examples of spam and non-spam messages for testing purposes.

## Steps

1. **Data Preparation** :

* The dataset `SMSSpamCollection.csv` is loaded and processed. This dataset contains labeled SMS messages categorized as either "spam" or "ham".

2. **Feature Engineering** :

* The model uses a **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** to convert text messages into numerical vectors that represent the importance of words in each message.

3. **Model Selection** :

* A **LinearSVC (Linear Support Vector Classifier)** from `sklearn` is used as the classification model. The model is trained on the vectorized SMS messages.

4. **Prediction Function** :

* The function `sms_prediction` takes an input text message and returns a prediction indicating whether the message is spam or not spam.

5. **Interactive Gradio Interface** :

* The project utilizes **Gradio** to build a simple web interface that allows users to input text messages and receive real-time predictions about whether the message is spam or not spam.
* The interface includes labeled input and output textboxes and a submit button to display the prediction.

## Dataset

The dataset used in this project is the `SMSSpamCollection.csv`, which contains SMS messages labeled as either "spam" or "ham" (not spam). This dataset is used to train the classifier.

![](images/df.png)

## Model

* **Classifier** : LinearSVC
* **Vectorizer** : TF-IDF Vectorizer
* **Pipeline** : The model is wrapped in a Scikit-learn pipeline for ease of use and integration with Gradio.

## Code Breakdown

##### SMS Classification Function

Takes `sms_text_clf` test data, sets X (feature), y (target data), splits data into training and test sets, and returns the `text_clf` object, which is a `Pipeline` object.

The pipeline streamline the series of data processing steps including:

* **TF-IDF Vectorization** : converts the text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF), which helps in transforming the text into a format that the machine learning model can understand `stop_words='english'` parameter removes common English words that are unlikely to be useful for classification.
* **Linear Support Vector Classification (SVC)** : a Support Vector Machine (SVM) classification model with a linear kernel, that has been trained on classifier that has been trained on the transformed training data and is used to classify the text messages

  ![](images/pipeline.png)

NOTE: *Function  `sms_classification` returns a fitted pipeline model for SMS classification (`text_clf` object, which is a `Pipeline` ) that includes the TF-IDF Vectorization to convert text messages into numerical features and a Linear Support Vector Classification (SVC) classification model, trained on the transformed training data.*

##### SMS Prediction Function

The `sms_prediction` function takes the text iterable (see below) and predict the classifications of text as SPAM or not.

It calles the Pipline predict method, then returns a message indicating whether the text is spam or not, based on the prediction of the classifier model.

    `prediction = text_clf.predict([text])`

* *Data Transformation: The pipeline includes steps like TfidfVectorizer, which are designed to transform a collection of text documents into numerical features.*

## Example Usage

Once the app is running, you can enter a message into the textbox and click 'Submit' to get a prediction. The model will classify the message as either "spam" or "not spam".

Sample input examples:

* Spam message: "Congratulations! You've won a $1,000 gift card. Click here to claim your prize."
* Non-spam message: "Hey, don't forget about the meeting tomorrow at 10 AM."

Sample input/output examples:

![img](https://file+.vscode-resource.vscode-cdn.net/c%3A/ai_projects/sms_spam_detector/images/output_non.png)

## Testing

There are pre-defined test cases in the project including:

* Spam message examples.
* Non-spam message examples.
* Non-spam texts in a casual "Gen Z" style for diversity in testing.

## Future Enhancements

* Add more robust pre-processing to handle special characters and emojis.
* Train the model on a larger dataset for improved performance.
* Explore deep learning methods for text classification.

## License

This project is licensed under the MIT License.
