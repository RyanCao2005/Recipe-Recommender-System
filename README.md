# Introduction
Recipes and ratings play a central role in shaping our culinary experiences, guiding how we discover and enjoy food. Predicting user preferences is essential for creating personalized and effective recipe recommendations. This project leverages machine learning to predict recipe ratings using numerical and textual features from the dataset. To address the challenge of imbalanced ratings, we employ advanced techniques such as TF-IDF, unsupervised learning methods like PCA, and ensemble learning techniques using Random Forest for robust multi-class classification.

From a practical standpoint, this work lays the groundwork for building recommender systems that align more closely with individual tastes, even when working with messy or imbalanced datasets. Theoretically, it offers valuable insights into applying machine learning to real-world challenges, particularly in managing skewed data distributions and improving predictive accuracy.

### Central Question and Importance

**How can we robustly and accurately predict the rating of a recipe based on its attributes, despite the ratings being heavily skewed towards 5/5?**

This question is crucial for building a recommender system that not only predicts ratings but also identifies the factors behind user satisfaction. It also addresses the broader challenge of handling imbalanced data, a common issue in many real-world datasets. By solving this, we gain insights into creating personalized systems across various domains while developing effective strategies for managing heavily skewed datasets.
### Dataset Details
- **Number of Rows**: The dataset consists of **220373** rows.
- **Relevant Columns**:
  placeholder for the real columns

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
## Exploratory Data Analysis and Data Cleaning

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
Our rationale behind these thresholds is that after inspecting the outliers we realized that they included wholesale.

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



## Bivariate Analysis

The box plot below shows the relationship between the number of steps in a recipe (`n_steps`) and the recipe's rating (`rating`). The plot reveals that, on average, recipes with higher ratings tend to have more steps, with some variations in the number of steps across different ratings. Notably, 5-star recipes exhibit a wide range of steps, but the median number of steps appears consistent, suggesting that well-rated recipes may vary in complexity but still maintain a high level of user satisfaction.

<iframe
  src="assets/distribution_of_ratings_n_steps.html"
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






