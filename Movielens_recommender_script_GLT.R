##################################
###### 1.A Data preparation ######
##################################

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# The following code splits the edx database into train and test:

set.seed(1)
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-edx_test_index,]
edx_test <- edx[edx_test_index,]


# We will then check that userId and movieId in the edx_edx_test set are also in the edx_edx_train, 
# and add the removed movieIds back into the edx_edx_train:

edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

removed <- edx_test %>% 
  anti_join(edx_train, by = "movieId")

edx_train <- rbind(edx_train, removed)

#################################
###### 1.B Data exploration ######
#################################

# Structure of the dataset:
str(edx)
head(edx)

# Check the dataset for missing values:
summary(edx)


# Number of unique users and movies:
edx %>%
  summarise(n_users = n_distinct(userId), n_movies = n_distinct(movieId)) %>%
  print.data.frame()


# Count of the star ratings:

edx %>% count(rating) %>% arrange(desc(n))


# Histogram of the ratings:
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black", fill = "blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")



# We observe large variations in the number of movies rated by each user:
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "blue") + 
  scale_x_log10() + 
  xlab("Number of ratings") +
  ylab("Number of users") +
  ggtitle("Number of ratings per user")



# Mean rating (only users with at least 100 ratings):
edx %>% 
  group_by(userId) %>% 
  filter(n() >= 100) %>%
  summarise(mean_rating = mean(rating)) %>% 
  ggplot(aes(mean_rating)) + 
  geom_histogram(bins = 30, color = "black", fill = "blue") +
  xlab("Mean movie rating") +
  ylab("Number of users") +
  ggtitle("Mean movie rating per user") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5)))


# Number of ratings per movie:
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "blue") + 
  xlab("Number of ratings") +
  ylab("Movie count") +
  scale_x_log10() + 
  ggtitle("Number of ratings per movie")



# Number of movies with a single rating:
edx %>% 
  group_by(movieId) %>%
  summarise(count = n()) %>% 
  filter(count == 1) %>% count(count) 


###########################################
###### 2. Data analysis and modelling ######
###########################################


## 2.A MODEL-BASED APPROACH ##
#############################

# This is a very simple rating model under which we we predict the same rating for all movies regardless of user. 
# This model is based on the average of all ratings in the training dataset and will serve as the baseline for successive models.

# The following code calculates our the mean rating (mu) in the train dataset:
mu <- mean(edx_train$rating)
mu


#The following code generates a function that will calculate the RMSE for actual values (true_ratings) from our test set to their corresponding predictors from our models:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# The RMSE function computes the root mean squared error between actual and predicted ratings (mu) and will be the metric by which we will evaluate the model. 
baseline_rmse <- RMSE(edx_test$rating,mu)
baseline_rmse

# Let's create a results table with this baseline model approach:
rmse_results <- data_frame(Method = "Baseline", RMSE = baseline_rmse)
rmse_results %>% knitr::kable()

## 2.B MOVIE EFFECT ##
######################

# Are some movies consistently rated higher or lower than others? To improve upon the baseline, we will build a model that computes the movie bias (b_i), 
# i.e. the deviation of each movie's mean rating from the total mean of all movies "mu". 
# This is a way of taking into account the inherent value/popularity of the movie.
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))


# The new predicition takes into account the fact that different movies are rated differently by adding the movie bias (b_i) to the total mean of all movies (mu).

predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

Movie_effect_rmse <- RMSE(predicted_ratings, edx_test$rating)

rmse_results <- bind_rows(rmse_results, data_frame(Method = "Movie Effect Model", RMSE = Movie_effect_rmse))
rmse_results %>% knitr::kable()

## 2.C MOVIE + USER EFFECT ##
####################################

# We can expect that movie ratings vary considerably not only depending on the movie, but also on the users.
# Let's see if including a user effect into the model will result into an improved RMSE.

# The following code computes the user effect, which we will call "b_u".
user_avgs <- edx_train %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# We predict ratings taking into account "b_i" and "b_u".
predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

User_effect_RMSE <- RMSE(predicted_ratings, edx_test$rating)
User_effect_RMSE

rmse_results <- bind_rows(rmse_results, data_frame(Method = "Movie & User Effects Model", RMSE = User_effect_RMSE))
rmse_results %>% knitr::kable()

#Top 10 best movies from the Movie Effect model
edx %>% left_join(movie_avgs, by="movieId") %>% group_by(title,movieId, b_i) %>% summarise(count=n()) %>% arrange(desc(b_i))  %>% head(.,10) %>% knitr::kable()

#Top 10 worst movies from the Movie Effect model
edx %>% left_join(movie_avgs, by="movieId") %>% group_by(title,movieId, b_i) %>% summarise(count=n()) %>% arrange(b_i)  %>% head(.,10) %>% knitr::kable()


## 4. REGULARIZED MOVIE-EFFECT APPROACH ##
###########################################

# Another factor to consider is that the number of times a movie is rated affects significantly the estimates of the "b_i", negatively or positively.
# Let's have a look at how this applies to our dataset:


movie_titles <- edx_train %>% 
  select(movieId,title) %>%
  distinct()

# Top 10 rated movies according to the prediction:

edx_train %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Worst 10  movies according to the prediction:

edx_train %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Most movies with a large b_i, positive or negative, are indeed rated a limited number of times.
# To control the total variability of the movie effects, we will use regularization, 
# i.e. we will apply a penalty lambda to movies that have been rated only a few times.
# Lambda is a parameter which can be optimized. We will use cross-validation 
# and compute RMSEs for different penalties (from 0 to 10 at a sequence of .25) applied to the 
# movie bias. We can thus determine the optimum lambda, i.e. the one that will minimize the RMSE.

lambdas <- seq(0, 10, 0.25)

mu <- mean(edx_train$rating)
just_the_sum <- edx_train %>% 
  group_by(movieId) %>% 
  summarise(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- edx_test %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]


# The lowest lambda according to this calculation is 1.5. Let's compute the regularized estimates of"b_i" using this value.

lambda <- 1.5
mu <- mean(edx_train$rating)
movie_reg_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarise(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# This graph shows how the estimates shrink with the penalty:
data_frame(original = movie_avgs$b_i, 
           regularized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#This code now shows the top 10 movies after regularization. Most are well-known titles which have been rated several times.
edx_train %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# And these are the movies rated lowest after regularization: 
edx_train %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


#The following code creates our predictions for movie bias applying Regularization:
predicted_ratings <- edx_test%>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

#The following code measures how well our predictions performed on the test set and then adds the RMSE to our results table:
Regularized_movie_effect_rmse <- RMSE(predicted_ratings, edx_test$rating)
Regularized_movie_effect_rmse

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Regularized Movie Effect Model",  
                                     RMSE = Regularized_movie_effect_rmse ))
rmse_results %>% knitr::kable()

# The penalized estimates provide a slight improvement over the least squares estimated above.



## 4. REGULARIZED MOVIE + USER EFFECT APPROACH ##
#################################################

# To improve our RMSE, let's include a regularized user effect into the model above.

lambdas <- seq(0, 10, 0.25)

# We will use sapply to join the lambdas with a function which applies lambda to 
# the user and movie bias. The following code will return the result of how our predictions compared to edx_test.

rmses <- sapply(lambdas, function(l){
  
  #start with the mean
    mu <- mean(edx_train$rating)
    
  #add the regularized movie effect
    
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  #add the user effect  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, edx_test$rating))
})

qplot(lambdas, rmses) 

# This code tells us the lambda for the lowest RMSE
lambda <- lambdas[which.min(rmses)]
lambda



# Measure how well our predictions performed on the edx_test set and then adds the RMSE to our results table:
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Regularized Movie + User Effect Model - test dataset",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

############################################################################
# Final RMSE value of the predicted ratings against the validation dataset #
############################################################################

# Calculate b_i with lambda penalty term:
b_i <- edx %>% group_by(movieId) %>% summarise(b_i = sum(rating - mu)/(n()+lambda))

# Calculate b_u with lambda penalty term:
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u = sum(rating - b_i - mu)/(n()+lambda))

#predict ratings with the validation set:
predicted_ratings <- validation %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% .$pred

# Calculate RMSE:
validation_rmse <- RMSE(predicted_ratings, validation$rating)

# Append the results to rmse_results:
rmse_results <- bind_rows(rmse_results,data_frame(Method = "Regularized Movie + User Effect Model - validation dataset",RMSE = validation_rmse))
rmse_results %>% knitr::kable()




