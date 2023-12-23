
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

# Note

- Downloaded the data from Kaggle and imported it as a zip file.

# Build the Flow

- Click Flow Actions at the bottom right of the Flow.

- Click Build all.

- Keep the default settings and click Build.

# Explored the data
The Flow contains two data sources:

IMDB_train : Training data to create the model.

IMDB_test : Testing data to estimate of the model’s performance on data that it did not see during training.

### Started by exploring the prepared training data.

1. Open the IMDB_train_prepared dataset.

2. Computed the number of records.

3. Analyze the columns. In the Categorical tab, recognized a small number of duplicates.

4. Withiun the Natural Language tab. Founnd that the most frequent words include movie, film, and br (the HTML tag for a line break) by using the "compute" feature.

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

Let’s try out one of the most frequently-used steps.

Return to the Flow, and open the Prepare recipe that produces the IMDB_train_prepared dataset.

Open the dropdown for the text column.

Select Simplify text.

Select the options to Stem words and Clear stop words.

Click Run, and open the output dataset.

Dataiku screenshot of the simplify text processor in a Prepare recipe.
Important

Dataiku suggested the Simplify text processor because it auto-detected the meaning of the text column to be natural language.

Train a new session of models
With a simplified text column, let’s train a new session of models.

Return to the previous modeling task. You can find it in the Lab tab of the right sidebar of the output dataset or the Visual Analyses page (G + A).

Click Train.

Name the session cleaned.

Click Train again.

Dataiku screenshot of a dialog for training a model.
Important

In the training dialog, Dataiku alerts us that the training data is different from the previous session.

In this case, cleaning the text led to only very minimal (if any) improvements over the baseline effort. This does not mean, however, that this was not a worthwhile step. We still significantly reduced the feature space without any loss in performance.

Dataiku screenshot of the results of a second model.
Formulas for feature generation
Next, let’s try some feature engineering.

Return to the Prepare recipe.

Click + Add a New Step at the bottom left.

Search for and select Formula.

Name the output column length.

Copy-paste the expression length(text) to calculate the length in characters of the movie review.

Click Run, and open the output dataset again.

Dataiku screenshot of a formula step in a Prepare recipe.
Tip

You can use the Analyze tool on the new length column to see its distribution from the Prepare recipe. You could also use the Charts tab in the output dataset to examine if there is a relationship between the length of a review and its polarity.

Train more models
Once again, let’s train a new session of models.

Return to the previous modeling task.

Before training, navigate to the Design tab.

Select the Features handling panel on the left.

Confirm length is included as a feature.

Click Train.

Name the session length added.

Click Train again.

Dataiku screenshot of a dialog for training a model.
This iteration shows a bit more improvement in both the logistic regression and random forest models.

Dataiku screenshot of the results of a third model.
Aside from the specific results we witnessed here, the larger takeaway is the value that text cleaning and feature engineering can bring to any NLP modeling task.

Tip

Feel free to experiment by adding new features on your own. For example, you might use:

A formula to calculate the ratio of the string length of the raw and simplified text column.

The Extract numbers processor to identify which reviews have numbers.

The Count occurrences processor to count the number of times some indicative word appears in a review.

Pre-process text features for machine learning
We now have applied simple text cleaning operations and feature engineering to improve a model that classifies positive and negative movie reviews.

However, these are not the only tools at our disposal. Dataiku also offers many preprocessing methods within the model design.

Let’s experiment with different text handling methods before we evaluate the performance of our chosen model against a test dataset.

Count vectorization text handling
Let’s apply count vectorization to the text feature.

From the modeling task, navigate to the Design tab.

Select the Features handling pane.

Select the text feature.

Change the text handling setting to Count vectorization.

Click Train.

Name the session count vec.

Read the warning message about unsupported sparse features, and then check the box to ignore it.

Click Train.

Dataiku screenshot of the features handling panel of a model.
When the fourth session finishes training, we can see that the count vectorization models have a performance edge over their predecessors.

In addition to the performance boost, they also have an edge in interpretability. In the logistic regression model for example, we can see features such as whether the text column contains a word like worst or excel.

On the left, select the Logistic Regression model from the count vec session.

Navigate to the Regression coefficients panel on the left.

Dataiku screenshot of the regression coefficients of a model.
These benefits, however, did not come entirely for free. We can expect the training time of the count vectorization models to be longer than that of the tokenization, hashing and SVD models.

For this small dataset, the difference may be relatively small. For a larger dataset, the increase in training time could be a critical tradeoff.

To see for yourself:

Click Models near the top left to return to the model training results.

Click the Table view on the right to compare training times.

Dataiku screenshot of a table of model results.
Tip

On your own, try training more models with different settings in the feature handling pane. For example:

Switch the text handling method to TF/IDF vectorization.

Observe the effects of increasing or decreasing the “Min. rows fraction %” or “Max. rows fraction %”.

Include bigrams in the Ngrams setting by increasing the upper limit to 2 words.

Deploy a model from the Lab to Flow
When you have sufficiently explored building models, the next step is to deploy one from the Lab to the Flow.

A number of factors — such as performance, interpretability, and scalability — could influence the decision of which model to deploy. Here, we’ll just choose our best performing model.

From the Result tab of the modeling task, select the logistic regression model from the count vec session.

Click Deploy near the upper right corner.

Click Create.

Dataiku screenshot of the dialog to deploy a model from the Lab to the Flow.
Copy data preparation steps
With a model now in the Flow, the relevant question is whether this model will perform as well on data that it has never faced before. A steep drop in performance could be a symptom of overfitting the training data.

The Evaluate recipe can help answer this question. But first, we need to ensure that the test data passes through all of the same preparation steps that the training data received.

We’ll start by copying the existing Prepare recipe.

Select the Prepare recipe in the Flow.

In the Actions sidebar, click Copy.

On the left, change the input dataset to IMDB_test.

Name the output dataset IMDB_test_prepared.

Click Create Recipe.

In the recipe’s first step, delete the non-existent sample column so that the step only removes the sentiment column.

Click Run.

Dataiku screenshot of the dialog for copying a Prepare recipe.
Evaluate the model
Once the test data has received the same preparation treatment, we are ready to test how the model will do on this new dataset.

Let’s take the first step toward this goal.

From the Flow, select the deployed model Predict polarity (binary) and the IMDB_test_prepared dataset.

Select the Evaluate recipe from the Actions sidebar.

Click Set for the output dataset. Name it IMDB_scored, and click Create Dataset.

Click Create Recipe.

Click Run on the Settings tab of the recipe, and then open the output dataset.

Dataiku screenshot of the dialog for an Evaluate recipe.
The Evaluate recipe has appended class probabilities and predictions to the IMDB_scored dataset. As we update the active version of the model, we could keep running the Evaluate recipe to check the performance against this test dataset.

Overall, it appears that the model’s performance on the test data was very similar to the performance on the training data. One way you could confirm this is by using the Analyze tool on the prediction and prediction_correct columns.


The Evaluate recipe includes options for two other types of output.

One is a dataset of metric logs comparing the performance of the model’s active version against the input dataset. It’s particularly useful for exporting to perform analysis elsewhere, such as perhaps a webapp.

The second is a model evaluation store, a key tool for evaluating model performance in Dataiku, but outside our scope here.

# Thank you
Viola! I have successfully built a model to perform a binary classification of the text sentiment. Along the way, I:

- Witnessed the importance of text cleaning and feature engineering.

- Explored the tradeoffs between different text handling strategies.

- Evaluated model performance against a test dataset.