# Truth-Tracker
This project builds a fake‑news detection system using a Bidirectional LSTM (BiLSTM) combined with a scikit‑learn classifier on a Kaggle‑style fake‑vs‑real news dataset. Text preprocessing steps such as cleaning, lowercasing, removing URLs and stopwords are applied before training, and the model pipeline is trained to classify articles as fake or real while evaluated using scikit‑learn metrics like accuracy, precision, recall, and F1‑score. The system supports live prediction on new articles or user‑input text through a backend API, with models saved and reloaded for reuse, and also includes a simple web‑based dashboard where users can submit news and instantly see the prediction.

**Pipeline Overview**

**1. Dataset and Initial Setup**
The pipeline starts with a Kaggle‑style fake‑vs‑real news dataset containing news text (title or full article) and binary labels (fake or real). The dataset is split into training and testing sets using stratified sampling to maintain label balance, and basic statistics are logged to understand text length, class distribution, and overall data quality before training.

**2. Text Preprocessing and Feature Preparation**
Raw text is cleaned by converting to lowercase, removing HTML, extra whitespace, URLs, and special characters. Stopwords are filtered out, and the remaining text is tokenized. Two parallel feature streams are prepared:

  i. Token sequences for the BiLSTM model (using Keras tokenization and padding).TF‑IDF vectors for the scikit‑learn classifier.
  ii. This ensures both deep and classical models can operate on the same high‑quality input.

**3. BiLSTM Model Design and Training**
A Bidirectional LSTM (BiLSTM) model is built using Keras/TensorFlow. It consists of an embedding layer followed by a BiLSTM layer and a dense classification head. The model is trained on the token‑sequence representation of the training set, learning contextual patterns in both forward and backward directions to capture subtle cues that distinguish fake from real news.

**4. Scikit‑learn Classifier Integration**
The BiLSTM’s outputs (e.g., hidden states or prediction probabilities) or TF‑IDF features are fed into a scikit‑learn classifier such as Logistic Regression or Random Forest. This hybrid design combines the BiLSTM’s contextual understanding with the calibration and interpretability of classical ML, improving overall robustness and stability.

**5. Evaluation and Metrics Calculation**
After training, both components are evaluated on the test set. Predictions are compared against ground‑truth labels and metrics such as accuracy, precision, recall, and F1‑score are computed using scikit‑learn. These metrics are logged for every training run, enabling performance tracking and model iteration.

**6. Model Saving and API Integration**
The trained BiLSTM and scikit‑learn model are saved to disk (e.g., .h5 and .pkl) for reuse. A lightweight REST API (built with Flask/FastAPI) is exposed, which accepts user‑submitted text via a POST /check_news endpoint. The API preprocesses the input, runs the BiLSTM–scikit‑learn pipeline, and returns a structured JSON response with the predicted label (fake/real) and confidence score.

**7. Live Prediction and Dashboard**
A simple web dashboard is implemented where users can paste news text or articles. The dashboard sends the text to the backend API, receives the model’s prediction, and displays it in real time. This completes the pipeline from raw dataset to a practical, user‑facing fake‑news detection system.
