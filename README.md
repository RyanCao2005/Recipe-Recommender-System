Made by Suchit Bhayani and Ryan Cao

# Introduction
Recipes and ratings play a central role in shaping our culinary experiences, guiding how we discover and enjoy food. Predicting user preferences is essential for creating personalized and effective recipe recommendations. This project leverages machine learning to predict recipe ratings using numerical and textual features from the dataset. To address the challenge of imbalanced ratings, we employ advanced techniques such as TF-IDF, unsupervised learning methods like PCA, and ensemble learning algorithms like Random Forest for robust multi-class classification.

From a practical standpoint, this work lays the groundwork for building recommender systems that align more closely with individual tastes, even when working with messy or imbalanced datasets, and by minimizing computational costs. From a theoretical standpoint, it offers valuable insights into applying machine learning to real-world challenges, particularly in managing skewed data distributions, and improving predictive accuracy.

### Central Question and Importance

**How can we robustly and accurately predict the rating of a recipe based on its attributes, despite the ratings being heavily skewed towards 5/5?**

This question is crucial for building a recommender system that not only predicts ratings but also identifies the factors behind user satisfaction. It also addresses the broader challenge of handling imbalanced data, a common issue in many real-world datasets. By solving this, we gain insights into creating personalized systems across various domains while developing effective strategies for managing heavily skewed datasets.
### Dataset Details
- **Number of Rows**: The dataset consists of **220373** rows.
- **Relevant Columns**: The dataset consists of **12** relevant columns

## Data Description

The project utilizes two primary datasets: the **recipes dataset** and the **interactions dataset**. Below are the details of the raw data frames:

### Recipes Dataset
This dataset , `recipe`, contains 83782 rows, which indicates 83782 unique recipes, and 12 columns which give detailed information about individual recipes:

| Column         | Description                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------|
| `name`         | The name of the recipe.                                                                      |
| `id`           | Unique ID for each recipe.                                                           |
| `minutes`      | Time required to prepare the recipe, in minutes.                                             |
| `contributor_id` | User ID of the person who submitted the recipe.                                             |
| `submitted`    | Date the recipe was submitted.                                                           |
| `tags`         | Tags assigned to the recipe (e.g., "vegan," "dessert").                                      |
| `nutrition`    | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV = “percentage of daily value” |
| `n_steps`      | Total number of steps involved in preparing the recipe.                                      |
| `steps`        | Ordered list of instructions for making the recipe.                                          |
| `description`  | User-provided description of the recipe.                                                     |
| `ingredients`  | ingredient needed to make recipe in the form of a list of text
| `n_ingredients`| Total number of ingredients used in the recipe.                                              |

### Interactions Dataset
This dataset , `interactions`, contains 731927 rows and 5 columns that capture user interactions with recipes, including reviews and ratings:

| Column     | Description                                      |
|------------|--------------------------------------------------|
| `user_id`  | Unique ID for each user.                |
| `recipe_id`| Unique ID for each recipe.              |
| `date`     | Date of interaction.             |
| `rating`   | Rating given by the user (1 - 5).              |
| `review`   | Text review provided by the user.               |



To create a unified dataset, the **recipes dataset** was merged with the **interactions dataset** using the `id` column from the recipes dataset and the `recipe_id` column from the interactions dataset. This step allowed us to align recipe details with user interactions, forming a comprehensive view of recipe ratings and reviews.

Both datasets provide essential information for this project. The recipes dataset offers key features such as `nutrition` and `minutes` (used for numerical analysis) and categorical columns like `ingredients`, `name`, and `description`. The interactions dataset helps in understanding user preferences through ratings and reviews. 
## Relevant Columns

During data preprocessing, we dropped several columns from the recipes and interactions datasets that were not directly relevant to our modeling goals. Below is a summary of the dropped columns and their descriptions:

### Dropped Columns from Recipes Dataset

| Column           | Description                                                                           |
|------------------|---------------------------------------------------------------------------------------|
| `contributor_id` | User ID of the person who submitted the recipe.                                       |
| `submitted`      | The date the recipe was submitted.                                                   |
| `id`             | Unique identifier for each recipe.                                                   |
| `name`           | The name of the recipe.                                                              |
| `tags`           | Tags assigned to the recipe (e.g., "vegan," "dessert").                              |
| `description`    | User-provided description of the recipe.                                             |
| `steps`          | Ordered list of instructions for making the recipe.                                  |

### Dropped Columns from Interactions Dataset

| Column     | Description                                      |
|------------|--------------------------------------------------|
| `user_id`  | Unique identifier for each user.                |
| `recipe_id`| Unique identifier for each recipe.              |
| `date`     | Date when the interaction occurred.             |

These columns were excluded either because they were not informative for our prediction task or because they duplicated information already captured by other columns. By dropping these, we streamlined the dataset to focus on the most relevant features for building our predictive models.

## Relevant Columns

For this project, we focused on retaining columns that provided the most value for predicting recipe ratings while dropping those that were redundant or irrelevant to our modeling goals. Below are the columns we kept and their descriptions:

### Kept Columns from Recipes Dataset

| Column         | Description                                                                             |
|----------------|-----------------------------------------------------------------------------------------|
| `minutes`      | Time required to prepare the recipe, in minutes.                                        |
| `nutrition`    | Nutrition details including calories, fat, sugar, sodium, protein, saturated fat, and carbs. |
| `n_steps`      | Total number of steps involved in preparing the recipe.                                 |
| `n_ingredients`| Total number of ingredients used in the recipe.                                         |

### Kept Columns from Recipes Dataset

| Column         | Description                                                                             |
|----------------|-----------------------------------------------------------------------------------------|
| `minutes`      | Time required to prepare the recipe, in minutes.                                        |
| `nutrition`    | Nutrition details including calories, fat, sugar, sodium, protein, saturated fat, and carbs. |
| `n_steps`      | Total number of steps involved in preparing the recipe.                                 |
| `n_ingredients`| Total number of ingredients used in the recipe.                                         |

### Dropped Columns
We excluded several columns from the recipes and interactions datasets, such as `contributor_id`, `submitted`, `user_id`, `recipe_id`, `date`, `description`, `id`, `name`, `tags`, and `steps`. These columns were removed because they were either redundant, irrelevant to the task, or did not contribute meaningfully to predicting user ratings.

By focusing on these key columns, we streamlined our dataset to maximize its predictive value while minimizing noise and unnecessary complexity.
# Exploratory Data Analysis and Data Cleaning

### Data Cleaning Pipeline

To prepare the dataset for analysis, we implemented a structured data cleaning pipeline that integrated the recipes and interactions datasets, addressed data inconsistencies, and extracted meaningful features. Below is an overview of the main steps:

#### Merging and Initial Cleaning
1. **Merge Datasets**: The recipes dataset was left-merged with the interactions dataset using the `id` column from recipes and the `recipe_id` column from interactions.
2. **Handle Zero Ratings**: Ratings of `0` were replaced with `NaN` since a zero rating is invalid and likely indicates a missing value. This reduced the skew caused by invalid ratings.
3. **Calculate Recipe Mean Ratings**: For recipes with multiple reviews, the average rating was calculated and added as a new column, providing an overall estimate of user preference for each recipe.

#### Transformations
1. **Extract Nutrition Information**: The `nutrition` column, initially stored as a string representing a list of metrics, was split into individual columns (e.g., calories, fat, sugar). This conversion enabled numerical analysis of nutritional values.
2. **Convert Strings to Lists**: Columns such as `tags`, `steps`, and `ingredients`, which were stored as strings of lists, were converted back into actual list formats for easier manipulation and analysis.
3. **Outlier Removal**: Domain knowledge was used to define thresholds for numerical columns:
   - Recipes exceeding **1,500 calories** or taking more than **360 minutes** to prepare were removed to reduce data skewness caused by extreme outliers.

#### Redundant Columns
Irrelevant or redundant columns, such as `contributor_id`, `submitted`, `name`,`id`,`recipe_id`, `user_id`,`tags`,`steps`, `date` and `description`, were dropped to minimize noise and improve model performance. These columns were either duplicative or did not contribute meaningfully to the predictive modeling task.


---

### Cleaned Dataframe Overview
After cleaning, the dataset retained key columns such as `rating`, `recipe mean rating`, `calories`, `minutes`, `n_steps`, `n_ingredients`, and processed list-based features like `tags`, `steps`, and `ingredients`. Below is a sample of the cleaned dataset:


|   minutes |   n_steps | ingredients                                                                                                                                                                    |   n_ingredients |   rating | review                                                                                                                                                                                                                                                                                                                                           |   recipe mean rating |   calories (#) |   total fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated fat (PDV) |   carbs (PDV) |
|----------:|----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------:|---------------:|------------------:|--------------:|---------------:|----------------:|----------------------:|--------------:|
|        40 |        10 | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour'] |               9 |        4 | These were pretty good, but took forever to bake.  I would send it ended up being almost an hour!  Even then, the brownies stuck to the foil, and were on the overly moist side and not easy to cut.  They did taste quite rich, though!  Made for My 3 Chefs.                                                                                   |                    4 |          138.4 |                10 |            50 |              3 |               3 |                    19 |             6 |
|        45 |        12 | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                    |              11 |        5 | Originally I was gonna cut the recipe in half (just the 2 of us here), but then we had a park-wide yard sale, & I made the whole batch & used them as enticements for potential buyers ~ what the hey, a free cookie as delicious as these are, definitely works its magic! Will be making these again, for sure! Thanks for posting the recipe! |                    5 |          595.1 |                46 |           211 |             22 |              13 |                    51 |            26 |
|        40 |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |        5 | This was one of the best broccoli casseroles that I have ever made.  I made my own chicken soup for this recipe. I was a bit worried about the tsp of soy sauce but it gave the casserole the best flavor. YUM!                                                                                                                                  |                    5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |             3 |
|           |           |                                                                                                                                                                                |                 |          | The photos you took (shapeweaver) inspired me to make this recipe and it actually does look just like them when it comes out of the oven.                                                                                                                                                                                                        |                      |                |                   |               |                |                 |                       |               |
|           |           |                                                                                                                                                                                |                 |          | Thanks so much for sharing your recipe shapeweaver. It was wonderful!  Going into my family's favorite Zaar cookbook :)                                                                                                                                                                                                                          |                      |                |                   |               |                |                 |                       |               |
|        40 |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |        5 | I made this for my son's first birthday party this weekend. Our guests INHALED it! Everyone kept saying how delicious it was. I was I could have gotten to try it.                                                                                                                                                                               |                    5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |             3 |
|        40 |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |        5 | Loved this.  Be sure to completely thaw the broccoli.  I didn&#039;t and it didn&#039;t get done in time specified.  Just cooked it a little longer though and it was perfect.  Thanks Chef.                                                                                                                                                     |                    5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |             3 |

### Outlier Filtering

To clean the dataset and ensure it reflects practical cooking scenarios, we identified and removed outliers in key numerical columns like `calories (#)` and `minutes`. These outliers likely represented rare or atypical scenarios that could distort analysis and model performance.

- **Calories Outliers**: Recipes with more than **1,500 calories** were removed. Our rationale assumes that users are primarily looking for healthier, single-serving recipes since they are cooking rather than ordering out. The presence of high-calorie outliers may be due to recipes designed for bulk meal preparation, where the nutritional metrics are aggregated for multiple servings, skewing the data.

- **Minutes Outliers**: Recipes with preparation times exceeding **360 minutes** (6 hours) were excluded. Such cases are rare in everyday cooking and likely represent bulk recipes or unusual situations, which can disproportionately affect trends in the dataset.

By filtering these outliers, we ensured that the dataset remains representative of typical user behavior and preferences. This step improves both the quality of insights derived from the data and the reliability of predictions made by our models.
## Univariate Analysis

The histogram below displays the distribution of recipe ratings on a scale of 1 to 5 stars. The data reveals a strong skew toward 5-star ratings, with the majority of recipes receiving top marks. This highlights the challenge of working with imbalanced data, where the high concentration of 5-star ratings may affect the performance of predictive models.

<iframe
  src="assets/distribution_of_ratings.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This insight is significant for the development of the recipe recommender system, as it suggests that most recipes are highly rated, but it also indicates the need for techniques to handle imbalanced datasets. By addressing this, the model can be made more robust and better equipped to recommend diverse recipes that may not always receive the highest ratings.

#### **Distribution of Calories**

The plot below shows the distribution of calories across recipes. This visualization provides insights into the caloric spread of recipes, highlighting whether most recipes are low or high in calories.

<iframe
  src="assets/distribution_of_calories.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot shows that the distribution is skewed to the right. Suggesting that there's a high concentration of low calorie recipes.

#### **Distribution of Number of Steps**

The plot below shows the distribution of the number of steps (`n_steps`) across recipes. Understanding this distribution helps us assess how complex or simple most recipes are.

<iframe
  src="assets/distribution_of_steps.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot shows that the distribution is skewed to the right. Suggesting that there's a high concentration of recipes that require a low number of steps to complete.

#### **Distribution of Number of Ingredients**

The plot below shows the distribution of the number of ingredients (`n_ingredients`) used in recipes. This visualization helps us understand the variability in ingredient requirements, which could influence recipe ratings.

<iframe
  src="assets/distribution_of_ingredients.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot shows that the distribution is skewed to the right. Suggesting that there's a high concentration of recipes that require a low number of ingredients.



## Bivariate Analysis

The box plot below shows the relationship between the number of steps in a recipe (`n_steps`) and the recipe's rating (`rating`). This plot suggests that rating is not affected by number of steps. Although it seems like this is the case we will see if this is actually true with further analysis.

<iframe
  src="assets/steps_violin_by_rating.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This visualization is significant for understanding how recipe complexity (as measured by the number of steps) influences user satisfaction. It can inform our recommender system by highlighting the balance between recipe difficulty and user ratings, helping to recommend recipes that are both achievable and highly rated.
## Interesting Aggregates

In this section, we examined the relationship between the number of steps (`n_steps`) in a recipe and the average cooking time (`minutes`). The data was grouped by the number of steps to analyze how cooking time varies with recipe complexity.

The plot below visualizes this relationship, showing how the average cooking time increases as the number of steps increases. As expected, more complex recipes with more steps tend to take more time, though some variation is observed, especially for recipes with a small number of steps.

<iframe
  src="assets/average_cooking_time_by_steps.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

# Missingness Assessment

## NMAR Analysis

In this section, we explore the potential for **Not Missing at Random (NMAR)** in the `review` column of our dataset. We hypothesize that the absence of a recipe description is not random, but rather influenced by certain factors. Specifically:

- **Popular Recipes**: If a recipe is already popular or highly rated, there may be less need for a detailed review, as users are likely already familiar with it. Therefore, the lack of a review may not be missing randomly, but because the recipe is well-known or widely recognized.
- **Simple Recipes**: Recipes that are simple to make, with few ingredients or steps, may also have missing reviews. In such cases, the recipe may not require an elaborate review, as the recipe will most likely be about a food that is not worth writing home about

This potential NMAR behavior in the `review` column introduces a bias in the data, as the absence of this information is likely tied to the recipe's characteristics rather than a random missingness process.

## Missingness Dependency

We moved on to examine the missingness of `rating` in the merged DataFrame by testing the dependency of its missingness. Specifically, we investigated whether the missingness in the `rating` column depends on the `n_ingredients` (number of ingredients) or `minutes` (cooking time) columns.

### Number of Ingredients and Rating

**Null Hypothesis**: The missingness of ratings does not depend on the number of ingredients in the recipe.

**Alternate Hypothesis**: The missingness of ratings does depend on the number of ingredients in the recipe.

#### Distribution of `n_ingredients` by Rating Missingness

The plot below shows the distribution of the number of ingredients (`n_ingredients`) for recipes where ratings are missing and not missing. This simplified horizontal violin plot highlights the spread and density of `n_ingredients` for each group, providing a clearer comparison.

<iframe
  src="assets/ingredients_violin_clean.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Permutation Test and Results

We ran a **permutation test** to assess whether the missingness of ratings is dependent on the number of ingredients in the recipe. The test statistic used was the **difference in means** of `n_ingredients` between the groups with missing ratings (`missing=True`) and non-missing ratings (`missing=False`).

The plot below shows the **empirical distribution of the test statistic** from the permutation test, with the observed test statistic highlighted in red.

<iframe
  src="assets/ingredients_histogram_clean.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

- **P-value**: **0.0**  
  This highly significant result provides strong evidence to reject the null hypothesis.

**Conclusion**: We conclude that the missingness of ratings **is dependent on the number of ingredients** in the recipe. Recipes with more ingredients are more likely to have missing ratings.

---

### Cooking Time and Rating

**Null Hypothesis**: The missingness of ratings does not depend on the cooking time of the recipe.

**Alternate Hypothesis**: The missingness of ratings does depend on the cooking time of the recipe.

#### Distribution of `minutes` by Rating Missingness

The plot below shows the distribution of recipe cooking time (`minutes`) for recipes where ratings are missing and not missing. This simplified horizontal violin plot highlights the spread and density of `minutes` for each group, providing a clearer comparison.

<iframe
  src="assets/minutes_violin_clean.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### Permutation Test and Results

We performed a **permutation test** to assess whether the missingness of ratings depends on the cooking time (`minutes`). The test statistic used was the **difference in means** of `minutes` between the groups with missing ratings (`missing=True`) and non-missing ratings (`missing=False`).

The plot below shows the **empirical distribution of the test statistic** from the permutation test, with the observed test statistic highlighted in red.

<iframe
  src="assets/minutes_histogram_clean.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

- **P-value**: **0.146**  
  This result is not statistically significant, so we do not reject the null hypothesis.

**Conclusion**: The missingness of ratings **is not dependent on the recipe's cooking time**. This indicates that cooking time does not significantly influence whether a rating is missing.

---

### P-value Calculation

The p-values for both tests were computed as the proportion of permuted test statistics greater than or equal to the observed test statistic. Based on these results, we conclude:
- The missingness of ratings **is dependent on the number of ingredients**.
- The missingness of ratings **is not dependent on the cooking time**.

### **Significance of Results**

The results of our permutation tests provide key insights into the missingness of ratings in relation to recipe attributes. We found that the missingness of ratings **is dependent on the number of ingredients**, with more complex recipes (those with more ingredients) being more likely to have missing ratings. This suggests that handling missing data for such recipes requires special attention. However, **cooking time** did not show a significant relationship with missingness, indicating that cooking time does not impact the likelihood of missing ratings. These findings are crucial for improving our recipe recommender system by better understanding the patterns of missing data and adjusting our modeling approach accordingly, ensuring higher accuracy in our model.

## **Hypothesis Testing: Difference in Steps Between Low and High Rated Recipes**

For this hypothesis test, we aimed to determine if there was a significant difference in the number of steps required for low and high-rated recipes. We were able to run this test by binarizing ratings as lower or high rated recipes where 3 was our threshold. We dropped columns with missing ratings for the sake of the test.

#### **Null Hypothesis**:
There is no difference in the distribution of steps among low and high-rated recipes.

#### **Alternative Hypothesis**:
Lower-rated recipes tend to have more steps than higher-rated recipes.

#### **Test Statistic**:
The test statistic used was the **difference in means** of the number of steps (`n_steps`) for lower-rated and higher-rated recipes. Specifically, we calculated the difference in the mean number of steps for **low-rated** recipes (rating 1-3) and **high-rated** recipes (rating 4-5).

#### **Significance Level**:
A significance level of **0.05** was used for this test.

#### **P-value**:
The p-value obtained from the permutation test was **0**.

#### **Conclusion**:

In our case, the p-value was **0**, which **is** statistically significant at the 0.05 level. Based on this, we **reject** the null hypothesis. This indicates that there is a **potential trend** suggesting that **lower-rated recipes may have more steps**.


---

### **Visualizing the Test Statistic Distribution**

Below is the **empirical distribution of the test statistic** from our permutation test. This visualization shows the distribution of the difference in means under the null hypothesis (after shuffling ratings) along with the observed test statistic (in red).

<iframe
  src="assets/number_of_steps_histogram_clean.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### **Justification of Choices**:

The **difference in means** is an appropriate test statistic because it directly compares the number of steps between low-rated and high-rated recipes. This helps us answer the question of whether the number of steps differs significantly between these two groups. A **permutation test** was chosen because it does not assume any specific distribution of the data, making it ideal for real-world data that might not follow traditional parametric assumptions. Using a significance level of **0.05** ensures a reasonable balance between Type I and Type II errors, which is standard in hypothesis testing. By focusing on **difference in means**(lower rated - rated higher), we are able to quantify the effect size and assess its statistical significance, providing clear insights into the relationship between recipe steps and rating.

# Prediction Problem

Our prediction problem is a **multi-class classification** task where we aim to predict the **rating** (response variable) of a recipe based on its attributes. Ratings are integers ranging from 1 to 5, and the goal is to classify recipes into one of these five rating categories.

### **Why We Chose the Response Variable**:
The rating reflects user satisfaction and is a key factor in building a personalized recommender system. Predicting this variable allows us to improve user experience by tailoring recipe recommendations to user preferences.

### **Evaluation Metric**:
We chose the **F1-score (macro-average)** as the primary evaluation metric for the following reasons:
1. **Class Imbalance**: The dataset is heavily skewed toward higher ratings (e.g., many 5-star ratings), which could cause metrics like accuracy to be misleading. Accuracy could be artificially high simply by predicting the majority class.
2. **Importance of Balanced Performance**: The macro-average F1-score computes the F1-score for each class and averages them equally, ensuring that all rating categories are weighted fairly, regardless of their prevalence.
3. **Focus on Precision and Recall**: F1-score balances precision (correctness of positive predictions) and recall (coverage of actual positives), which is crucial in scenarios where missing underrepresented classes could reduce the model’s usefulness.

### **Training Features and Justification**:
We selected features that would be available at the time of prediction. These include attributes such as `minutes`, `n_steps`, `n_ingredients`, `nutrition` details, and textual features like `ingredients` and `reviews`. These features are known before a recipe is rated and provide valuable information, such as preparation time, complexity, and potential sentiment from user-provided reviews.

# Baseline Model: Random Forest Classifier

### **Model Overview**
Our baseline model is a **Random Forest Classifier**, implemented using a pipeline that integrates feature preprocessing and hyperparameter tuning. The objective is to predict recipe ratings (multi-class classification) ranging from 1 to 5 based on recipe attributes.

### **Features**
The model uses the following features after preprocessing:
1. **Ingredients (`ingredients_highest_tfidf`)**:
   - **Nominal**: Encoded as the highest TF-IDF term for each recipe's list of ingredients.
2. **Review Text (`review_highest_tfidf`)**:
   - **Nominal**: Encoded as the highest TF-IDF term for each recipe's review text.
3. **Health Metrics**:
   - **Quantitative**: Several numerical health-related metrics about the recipes were included as-is without additional encoding:
     - `calories (#)`: Total calories in the recipe.
     - `total fat (PDV)`: Percentage daily value of total fat.
     - `sugar (PDV)`: Percentage daily value of sugar.
     - `protein (PDV)`: Percentage daily value of protein.
     - `saturated fat (PDV)`: Percentage daily value of saturated fat.
     - `carbs (PDV)`: Percentage daily value of carbohydrates.
4. **Other Numerical Features**:
   - `minutes`: Time required to prepare the recipe.
   - `n_steps`: Number of steps in the recipe.
   - `n_ingredients`: Number of ingredients in the recipe.

### **Encodings**
1. **TF-IDF Vectorization**:
   - Applied to `ingredients` and `review` columns to extract the highest-weighted TF-IDF term for each recipe. This captures the most important term describing the recipe’s content or sentiment.
2. **One-Hot Encoding**:
   - Applied to the resulting TF-IDF terms (`ingredients_highest_tfidf` and `review_highest_tfidf`) to convert them into a format suitable for machine learning.
3. **Passthrough for Numerical Features**:
   - Health metrics (`calories (#)`, `total fat (PDV)`, `sugar (PDV)`, etc.), as well as `minutes`, `n_steps`, and `n_ingredients`, were included as-is because they are already quantitative.

### **Hyperparameter Tuning**
The model's hyperparameters were tuned using **GridSearchCV** with 5-fold cross-validation. The parameters tuned were:
- `n_estimators`: Number of trees in the forest.
- `max_depth`: Maximum depth of each tree.
- `min_samples_split`: Minimum number of samples required to split an internal node.

### **Performance**
The model was evaluated using **5-fold cross-validation** on the training set. After hyperparameter tuning, the best-performing parameters were selected. On the test set:
- **F1-Score (Macro-Average)**: **0.1746**

### **Confusion Matrix**
The confusion matrix below illustrates the performance of the baseline model. Due to the highly imbalanced nature of the dataset, the model predicts the majority class (`5`) for almost all inputs, resulting in poor performance on minority classes.

<img src="assets/confusion_matrix_base.png" alt="Confusion Matrix for Baseline Model" width="600">

This visualization highlights the need for strategies to address class imbalance, such as oversampling minority classes, undersampling the majority class, or using class-weighted loss functions.

### **Conclusion**
The baseline model is a strong starting point for predicting recipe ratings, combining textual and numerical features effectively. However, it struggles with the dataset’s imbalanced nature, resulting in biased predictions. Addressing these imbalances and incorporating richer representations of textual data could significantly improve model performance.

# Final Model: Recipe Rating Prediction

### **Feature Engineering**
To improve upon the baseline model, we engineered new features that better represent the characteristics of recipes and their reviews:

1. **Imputed Ratings Based on Ingredients**:
   - Missing ratings were imputed probabilistically based on the number of ingredients (`n_ingredients`). This method assigns missing ratings by sampling from the existing ratings within the same `n_ingredients` group.
   - **Why it matters**: Recipes with similar complexity (as reflected by the number of ingredients) are likely to have comparable user feedback.

2. **PCA Analysis on TF-IDF Data**:
   - We applied **Principal Component Analysis (PCA)** to reduce dimensionality for features derived from `ingredients` and `review` columns after TF-IDF vectorization and One-Hot Encoding.
   - **Results**:
     - The variance explained by the components increased slowly as a function of the number of components, suggesting high intrinsic dimensionality.
     - **Conclusion**: PCA was not used in the final model as the benefits did not justify the computational cost.

Below are the plots of captured variance for `ingredients` and `review`:

#### PCA for Ingredients
<iframe
  src="assets/ingredients_pca_variance.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

#### PCA for Reviews
<iframe
  src="assets/review_pca_variance.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

### **Final Model Overview**
The final model is a **Random Forest Classifier**, chosen for its ability to handle categorical and numerical data effectively. We refined the model pipeline to incorporate the new engineered features while retaining the preprocessing steps from the baseline model.

1. **Pipeline Steps**:
   - TF-IDF vectorization for `ingredients` and `review` fields.
   - One-Hot Encoding for the highest TF-IDF terms.
   - Imputed ratings based on `n_ingredients`.
   - Standardized recipe metrics (`calories (#)`, `protein (PDV)`, etc.).
   - Passthrough on rest

2. **Hyperparameter Tuning**:
   - Hyperparameters were tuned using **GridSearchCV** with 3-fold cross-validation to optimize the performance of the **Random Forest Classifier**. The following types of hyperparameters were tuned:
   
     - **Number of Estimators**: Controls the number of trees in the random forest. A higher number of trees generally improves the model's performance but increases computational cost.
     
     - **Maximum Depth**: Defines the maximum depth of each decision tree. This hyperparameter controls the complexity of the trees and helps prevent overfitting.
     
     - **Class Weight**: Adjusts the weight of classes to handle imbalanced data. Options include methods that balance the weight of each class based on frequency or by sampling.

   - **Best Hyperparameter Combination**:  
     The best combination of hyperparameters found was:
     - `randomforestclassifier__n_estimators`: 500
     - `randomforestclassifier__max_depth`: 60
     - `randomforestclassifier__class_weight`: 'balanced_subsample'


### **Final Model Performance**
The final model was evaluated on the same test set as the baseline model to ensure a fair comparison. Key performance metrics include:

- **Baseline F1-Score (Macro-Average)**: **0.1746**
- **Final Model F1-Score (Macro-Average)**: **0.47174**

This improvement reflects the added value of our engineered features and hyperparameter tuning. However, the dataset's intrinsic complexity and high dimensionality remain challenges for achieving high accuracy.

---

### **Dummy Classifier Performance**
To further assess the performance of our final model, we also evaluated a **Dummy Classifier** with the `uniform` strategy, which randomly assigns labels to the test set. This provides a baseline comparison for evaluating whether the final model's improvements are significant.

- **Dummy Classifier F1-Score (Macro-Average)**: **0.11997**

The Dummy Classifier's performance is much lower than the final model, confirming that the final model's feature engineering and hyperparameter tuning provide a substantial improvement.

---

### **Confusion Matrix**
The following confusion matrix shows the performance of the final model on the test set:

<iframe
  src="assets/final_model_confusion_matrix.png"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Additionally, the confusion matrix for the Dummy Classifier is shown below for comparison:

<iframe
  src="assets/dummy_classifier_confusion_matrix.png"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---

### **Conclusion**

We engineered key features to enhance the recipe rating prediction model: imputed ratings based on the number of ingredients, which helped fill missing values in a meaningful way, and PCA analysis on TF-IDF data, which was explored but ultimately excluded due to computational constraints. The **Random Forest Classifier** was selected for its ability to handle both categorical and continuous features, with optimal hyperparameters found through **GridSearchCV**: 500 estimators, a maximum depth of 60, and 'balanced_subsample' class weight. These settings optimized model performance by capturing deep feature interactions and addressing class imbalances. The final model showed a significant improvement over the baseline, with the **F1-Score (Macro-Average)** rising from **0.1746** to **0.47174**. The Dummy Classifier, with an F1-Score of **0.11997**, confirmed the value of our engineered features and hyperparameter tuning. These improvements highlight the importance of meaningful feature engineering and model optimization in enhancing predictive accuracy.

# Fairness Analysis

### **Fairness Analysis by Calories**

We performed a fairness analysis to assess whether the model performs equally well on recipes with fewer than or equal to 1000 calories compared to those with more than 1000 calories. This is crucial for ensuring that the model provides balanced recommendations, especially for individuals seeking to bulk or gain weight. We compared the **F1-Score (Macro-Average)** between the two groups (low-calorie vs high-calorie) using a **permutation test**.

- **Null Hypothesis**: The model performs equally well for both groups (less than or equal to 1000 calories and greater than 1000 calories).
- **Alternative Hypothesis**: The model performs better for one group over the other.

The observed test statistic, which is the absolute difference in F1-scores between the two groups, was compared to the distribution of test statistics obtained from 1,000 permutations of the data. The **p-value** for the test was calculated as the proportion of permuted test statistics greater than or equal to the observed test statistic.

#### Results:
- The **p-value** from the fairness analysis was **0.128**.
- **Conclusion**: Since the p-value (0.128) is greater than our alpha level of 0.05, we **fail to reject the null hypothesis**. This means we do not have convincing evidence to suggest that our model is unfair towards higher calorie recipes. The model's performance on recipes with fewer or more than 1000 calories is comparable.

#### Test Statistic Distribution:
The histogram below visualizes the distribution of test statistics from the permutation test:

<iframe
  src="assets/fairness_test_statistics_histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The black line indicates the observed test statistic, which lies within the distribution of simulated test statistics, supporting the conclusion of fairness in model performance across calorie groups.
