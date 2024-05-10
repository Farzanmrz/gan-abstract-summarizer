# Amazon Recommender System Project

## Introduction

This notebook presents the construction and demonstration of a recommender system conceptualized within the framework of the AI-Informed Analytics Cycle. The system is engineered to address a significant query pertaining to consumer behavior and product interaction on the Amazon platform.

### Query

Central Inquiry: What electronic items from Amazon’s extensive inventory is a user predisposed to favor based on their historical rating data?

The resolution of this query provides value to both the individual consumers and the organization Amazon as explained below:

- **For the User (Shopper):** The system provides bespoke recommendations and facilitates product discovery congruent with individual preferences. This not only expedites navigability through numerous products but also boosts customer satisfaction. The value of this personalization can be empirically evaluated by observing user interaction with the suggested items which includes metrics such as click-through and conversion rates as well as the qualitative and quantitative feedback encapsulated in subsequent product reviews.

- **For the Organization (Amazon):** Sales are boosted because the system maximizes the probability of user engagement and purchase by providing users with personalized item recommendations. Increased sales act as direct contributors to the company’s fiscal throughput and also enhance customer loyalty and lifetime value. Additionally, the system’s ability to align product recommendations with consumer trends can lead to more efficient inventory management. The tangible value imparted to the organization can be quantitatively appraised by monitoring metrics such as the upsurge in sales volume, revenue increments attributable to the recommendation engine, and the optimization of inventory turnover ratios.

### Data or Knowledge Source

The system utilizes the Amazon 5-core Electronic items dataset specifically selected from the "Small subsets for experimentation" as outlined on UCSD’s dataset repository. This dataset, originally in JSON format and rich with various attributes, has been streamlined to focus on the essentials for our recommender system:

- **uid:** User ID
- **pid:** Product ID
- **rating:** User’s product rating (1.0 to 5.0)

To enhance efficiency and reduce notebook runtime, we preprocessed this dataset retaining only these critical variables and converting the data to CSV format. This tailored dataset emphasizing user preferences is ideal for developing our collaborative filtering-based recommender system. Given the same format, the recommender system could work for other categories besides electronics too.

The knowledge source for the construction and refinement of this recommender system was informed by the "Recommender Systems Handbook" (Ricci et al., 2015) during critical junctures of the design process. This book, along with the paper on recommendation systems in dating (Wobcke et al., 2015), offered valuable insights and methodological guidance instrumental in shaping key decision-making aspects of the system’s development.

## AI Complex Task

The system is tasked with the intricate AI challenge of forecasting user-item interactions that have yet to be observed. It is designed to accept the number of recommendations to generate alongside a list of users and subsequently produce a list containing recommended electronic items accompanied by their predicted ratings. For a practical demonstration and illustrative examples of inputs and outputs, please consult Section 6 of this notebook.

## AI Method

The recommender system is built using the surprise library, which provides tools for creating and analyzing recommender systems. Alongside libraries such as numpy for numerical operations, pandas for data handling, and visualization tools like matplotlib and seaborn are used for their robust data visualization capabilities.

We evaluate three collaborative filtering algorithms for our system:

- **SVD (Singular Value Decomposition):** This technique decomposes the user-item rating matrix to identify latent factors that capture the underlying patterns of ratings.
- **KNNBaseline:** A collaborative filtering algorithm that predicts ratings by considering the most similar users or items and incorporating overall rating tendencies.
- **BaselineOnly:** An approach that uses overall average ratings to predict a user’s rating for an item, taking into account the typical rating behavior of the user and the general reception of the item.

After testing these methods on a smaller dataset, SVD is chosen for further use due to its balance of performance and complexity. Hyperparameter tuning for the SVD model is performed using grid search with cross-validation to find the settings that yield the best prediction accuracy.

The Jupyter notebook can be run all at once, but the code blocks are building upon each other sequentially, so make sure to run them in order. A PDF version of this notebook with complete outputs is also provided to remove the necessity of having to run this notebook.

## References

- Ricci F., Rokach L., & Shapira B. (2015). *Recommender Systems Handbook* (Second). Springer-Verlag New York Inc.
- Wobcke W., Krzywicki A., Kim Y. S., Cai X., Bain M., Compton P., & Mahidadia A. (2015). A deployed people-to-people recommender system in online dating. *AI Magazine, 36*(3), 5–18. [https://doi.org/10.1609/aimag.v36i3.2599](https://doi.org/10.1609/aimag.v36i3.2599)
