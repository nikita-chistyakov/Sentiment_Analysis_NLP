
# Sentimet Analysis with NLP
The problem is a binary classification of text data: a sentiment analysis for movie reviews as positive or negative? My goal is to train a model on the text of a movie review that can predict the polarity of each review.

# Objectives

- Build a baseline binary classification model to predict text sentiment.

- Use text cleaning and feature generation methods to improve the performance of a sentiment analysis model.

- Experiment building models with different text handling preprocessing methods.

- Deploy a model from the Lab to the Flow and use the Evaluate recipe to test its performance against unseen data.

# Prerequisites
- Access to a Dataiku instance.

- Basic knowledge of visual ML in Dataiku.

# Creating the project
- From the Dataiku Design homepage, click + New Project > DSS tutorials > ML Practitioner > NLP - The Visual Way.

- From the project homepage, click Go to Flow (or G + F).

## Note

- Downloaded the data from Kaggle and imported it as a zip file.


# Explored the data
The Flow contains two data sources:

- IMDB_train : Training data to create the model.

- IMDB_test : Testing data to estimate of the model’s performance on data that it did not see during training.

### Started by exploring the prepared training data.

1. Opened the IMDB_train_prepared dataset.

2. Computed the number of records.

3. Analyzed the columns. In the Categorical tab, recognized a small number of duplicates.

4. Within the Natural Language tab. Founnd that the most frequent words include movie, film, and br (the HTML tag for a line break) by using the "compute" feature.

5. Analyzed the polarity column on whole data, and Compute metrics for the column to find an equal number of positive and negative reviews. This is helpful to know before modeling.

6. After finishing exploring IMDB_train_prepared, I followed a similar process to explore IMDB_test.


# Train a baseline model
With the training data ready, I've build an initial model to predict the polarity of a movie review.

## Created a quick prototype model
I started with a default quick prototype in the IMDB_train_prepared dataset.

- Performed an AutoML Prediction on polarity as the target variable for the prediction model. Created, and then Trained.

- The default text handling method is Tokenization, hashing and SVD. I’ll experiment with others later. For now, let’s stick with the default option.

# Train baseline models
With the text column now included as a feature in the model design, I trained a session of baseline models.

- Logistic regression (baseline) : 0.869
- Random forrest (baseline) : 0.810

# Cleaned text data
Now that I had a baseline model to classify movie reviews into positive and negative buckets, I wanted to try to improve performance with some simple text cleaning steps.

## Simplify text
The processor library includes a variety of steps for working with natural language data.

This is one of the most frequently-used steps.

In the Flow, the Prepare recipe produces the IMDB_train_prepared dataset.

- To simplify text, I selected the options to Stem words and Clear stop words.

- Created the output dataset.


# Trained a new session of models
I trained a new session of models with a simplified text column. 

- In this case, cleaning the text led to only very minimal (if any) improvements over the baseline effort. This does not mean, however, that this was not a worthwhile step. I still significantly reduced the feature space without any loss in performance.

### The results of a second model
- Logistic regression (baseline) : 0.869
- Random forrest (baseline) : 0.810


#### Next, I tried feature engineering

Steps for feature generation:

- Return to the Prepare recipe.
- Click + Add a New Step at the bottom left.
- Search for and select Formula.
- Name the output column length.
- Copy-paste the expression length(text) to calculate the length in characters of the movie review.
- Click Run, and open the output dataset again.


User note: I can use the Analyze tool on the new length column to see its distribution from the Prepare recipe. I can also use the Charts tab in the output dataset to examine if there is a relationship between the length of a review and its polarity.

# Train more models
Steps for training a new session of models:

- Return to the previous modeling task.
- Before training, navigate to the Design tab.
- Select the Features handling panel on the left.
- Confirm length is included as a feature.
- Click Train.
- Name the session length added.
- Click Train again.

This iteration shows a bit more improvement in both the logistic regression and random forest models.

Aside from the specific results I witnessed here, the larger takeaway is the value that text cleaning and feature engineering can bring to any NLP modeling task.


# Pre-process text features for machine learning
So far I have applying simple text cleaning operations and feature engineering to improve a model that classifies positive and negative movie reviews.

However, these are not the only tools at my disposal. Dataiku also offers many preprocessing methods within the model design.

Next, I am experimenting with different text handling methods before I can evaluate the performance of my chosen model against a test dataset.

## Count vectorization text handling
Steps for aplying count vectorization to the text feature:

- From the modeling task, navigate to the Design tab.
- Select the Features handling pane.
- Select the text feature.
- Change the text handling setting to Count vectorization.
- Click Train.
- Name the session count vec.
- Read the warning message about unsupported sparse features, and then check the box to ignore it.
- Click Train.

When the fourth session finishes training, I can see that the count vectorization models have a performance edge over their predecessors.

In addition to the performance boost, they also have an edge in interpretability. In the logistic regression model for example, I can see features such as whether the text column contains a word like worst or excel.

# 
These benefits, however, did not come entirely for free. I can expect the training time of the count vectorization models to be longer than that of the tokenization, hashing and SVD models.

For this small dataset, the difference may be relatively small. For a larger dataset, the increase in training time could be a critical tradeoff.


# Deploying a model from the Lab to Flow
When I have sufficiently explored building models, the next step is to deploy one from the Lab to the Flow.

A number of factors — such as performance, interpretability, and scalability — could influence the decision of which model to deploy. Here, I’ll just choose our best performing model.

From the Result tab of the modeling task, select the logistic regression model from the count vec session.

# Copying data preparation steps
With a model now in the Flow, the relevant question is whether this model will perform as well on data that it has never faced before. A steep drop in performance could be a symptom of overfitting the training data.

The Evaluate recipe can help answer this question. But first, I need to ensure that the test data passes through all of the same preparation steps that the training data received.

I’ll start by copying the existing Prepare recipe.

- Select the Prepare recipe in the Flow.
- In the Actions sidebar, click Copy.
- On the left, change the input dataset to IMDB_test.
- Name the output dataset IMDB_test_prepared.
- Click Create Recipe.
- In the recipe’s first step, delete the non-existent sample column so that the step only removes the sentiment column.
- Click Run.


# Evaluate the model
Once the test data has received the same preparation treatment, I was ready to test how the model will do on this new dataset.

My first step toward this goal:

- From the Flow, select the deployed model Predict polarity (binary) and the IMDB_test_prepared dataset.
- Select the Evaluate recipe from the Actions sidebar.
- Click Set for the output dataset. Name it IMDB_scored, and click Create Dataset.
- Click Create Recipe.
- Click Run on the Settings tab of the recipe, and then open the output dataset.

#
The Evaluate recipe has appended class probabilities and predictions to the IMDB_scored dataset. As I update the active version of the model, I could keep running the Evaluate recipe to check the performance against this test dataset.

#
Overall, it appears that the model’s performance on the test data was very similar to the performance on the training data. One way you could confirm this is by using the Analyze tool on the prediction and prediction_correct columns.


# Thank you
Viola! I have successfully built a model to perform a binary classification of the text sentiment. Along the way, I:

- Witnessed the importance of text cleaning and feature engineering.

- Explored the tradeoffs between different text handling strategies.

- Evaluated model performance against a test dataset.