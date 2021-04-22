##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

### End of Creating Data Sets ###
### Beginning of Creating Algorithm ###

# Split 'edx set' into training and test sets
set.seed(1, sample.kind="Rounding")
y <- edx$rating
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index]
test_set <- edx[test_index]

# Remove entries in test set for those not included in training set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Separate genres in training and test sets for 'genre effect' modeling
train_set_separated <- train_set %>%
  separate_rows(genres, sep = "\\|")

test_set_separated <- test_set %>%
  separate_rows(genres, sep = "\\|")

# Define RMSE (Root Mean Squared Error) function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Movie (M) Effects
mu_hat <- mean(train_set$rating) # This mu_hat is repeatedly used throughout this code
movie_avgs <- train_set %>%      # Contains movie to movie biases, b_i
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

bi_plot <- qplot(b_i, data = movie_avgs, bins = 20, color = I("black")) # plot showing b_i's vary substantially

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu_hat + b_i) %>%
  .$pred
M <- RMSE(test_set$rating, predicted_ratings) #> 0.94374

# Movie + User (MU) Effects
movie_user_avgs <- train_set %>%  # Contains user biases, b_u
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

bu_plot <- qplot(b_u, data = movie_user_avgs, bins = 30, color = I("black")) # plot showing b_u's vary substantially

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movie_user_avgs, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred
MU <- RMSE(test_set$rating, predicted_ratings) #> 0.86593

# Movie + User + Genre (MUG) Effects
genre_avgs <- train_set_separated %>%  # Contains genre biases, b_g
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movie_user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))

bg_plot <- qplot(b_g, data = genre_avgs, bins = 30, color = I("black")) # plot showing b_g's vary substantially

predicted_ratings <- test_set_separated %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movie_user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  .$pred

MUG <- RMSE(test_set_separated$rating, predicted_ratings) #> 0.86419

# Movie + User + Genre + Time (MUGT) Effects

  # Check if there is any evidence of time effect
date_plot <- edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>% 
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() #> plot shows there is some evidence, but not significant

  # Convert timestamp column into date and round it to weekly basis
train_set_date <- train_set_separated %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

test_set_date <- test_set_separated %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

time_avgs <- train_set_date %>%  # Contains time biases, b_t
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movie_user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu_hat - b_i - b_u - b_g))

predicted_ratings <- test_set_date %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movie_user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  left_join(time_avgs, by = "date") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g + b_t) %>%
  .$pred

MUGT <- RMSE(test_set_separated$rating, predicted_ratings) #> 0.86408

# Why We Need Regularized Model
movie_titles <- edx %>%  # For use below
  select(movieId, title) %>%
  distinct()

top10 <- train_set %>% 
  count(movieId) %>%  # Showing # of ratings for movies having top 10 large positive b_i's
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n) #> [1] 1 1 2 1 1 1 1 3 4 2

worst10 <- train_set %>% 
  count(movieId) %>%  # Showing # of ratings for movies having top 10 large negative b_i's
  left_join(movie_avgs, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n) #> [1] 2 1 1 1 30 40 161 10 2 1

# Regularized Movie + User (RMU) Effects

lambdas <- seq(0, 10, 0.25) # Use cross-validation to choose optimal lambda

rmses <- sapply(lambdas, function(d){
  
  mu_hat <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat) / (n() + d))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i) / (n() + d))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    .$pred
  
  return(RMSE(test_set$rating, predicted_ratings))
})

qplot(lambdas, rmses) 
lambda <- lambdas[which.min(rmses)] #> 4.75
RMU <- rmses[which.min(rmses)] #> 0.86524

# Regularized Movie + User + Genre (RMUG) Effects

lambda <- 4.75 # Verified it by using cross-validation but skipped it here
               # because it just takes too long to run the code

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat) / (n() + lambda))
  
b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i) / (n() + lambda))
  
b_g <- train_set_separated %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u) / (n() + lambda))
  
predicted_ratings <- test_set_separated %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  .$pred
  
RMUG <- RMSE(test_set_separated$rating, predicted_ratings) #> 0.86355

# Regularized Movie + User + Genre + Time (RMUGT) Effects

lambda <- 5 # Verified it by using cross-validation but skipped it here
            # because it just takes too long to run the code

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat) / (n() + lambda))

b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i) / (n() + lambda))

b_g <- train_set_separated %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u) / (n() + lambda))

b_t <- train_set_date %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - mu_hat - b_i - b_u - b_g) / (n() + lambda))

predicted_ratings <- test_set_date %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_t, by = "date") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g + b_t) %>%
  .$pred
  
RMUGT <- RMSE(test_set_separated$rating, predicted_ratings) #> 0.86339 (Lowest RMSE!)

# Comparison of all RMSE's (Bar Plot)
names <- c("MU", "MUG", "MUGT", "RMU", "RMUG", "RMUGT")
rmses <- round(c(MU, MUG, MUGT, RMU, RMUG, RMUGT), 5)
df <- as_tibble(cbind(names, rmses))
comparison <- df %>% 
  ggplot(aes(names, rmses, fill=ifelse(names=="RMUGT","A","B"))) +
  geom_col(show.legend=FALSE) +
  scale_fill_manual(values = c(A="firebrick3", B="steelblue4")) +
  labs(title = "RMSE's of All Models", x = "", y = "") +
  theme(plot.title = element_text(hjust = 0.4))

######################################################################
# Choose "Regularized Movie + User + Genre + Time Effects" Model
# Train the final model with edx data and test it on validation set
######################################################################

# edx set preparation
edx_separated <- edx %>%
  separate_rows(genres, sep = "\\|")

edx_date <- edx_separated %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

validation_separated <- validation %>%
  separate_rows(genres, sep = "\\|") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

lambda <- 5 

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat) / (n() + lambda))

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i) / (n() + lambda))

b_g <- edx_separated %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u) / (n() + lambda))

b_t <- edx_date %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - mu_hat - b_i - b_u - b_g) / (n() + lambda))

predicted_ratings <- validation_separated %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_t, by = "date") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g + b_t) %>%
  .$pred

RMSE(validation_separated$rating, predicted_ratings) #> 0.86249

