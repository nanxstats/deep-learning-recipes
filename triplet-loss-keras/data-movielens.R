read_movielens <- function() {
  url_uabase <- "http://files.grouplens.org/datasets/movielens/ml-100k/ua.base"
  url_uatest <- "http://files.grouplens.org/datasets/movielens/ml-100k/ua.test"
  train <- read.table(url_uabase)
  test <- read.table(url_uatest)
  names(train) <- names(test) <- c("user_id", "item_id", "rating", "timestamp")
  train$"timestamp" <- test$"timestamp" <- NULL
  list("train" = train, "test" = test)
}

binarize_ratings <- function(df) {
  idx_pos <- df$rating >= 4L
  idx_neg <- df$rating < 4L
  df$rating[idx_pos] <- 1L
  df$rating[idx_neg] <- 0L
  df
}

get_movielens_data <- function() {
  movielens <- read_movielens()
  train <- movielens$train
  test <- movielens$test

  # binarize the 1 to 5 star ratings
  train <- binarize_ratings(train)
  test <- binarize_ratings(test)

  list("train" = train, "test" = test)
}

# generate triplets
# (random non-positive item as the negative item)
get_triplets <- function(df) {
  # first, simply extract all positive pairs
  idx_pos <- which(df$rating == 1L)
  user_id <- df$user_id[idx_pos]
  item_id_pos <- df$item_id[idx_pos]

  df_ret <- data.frame(user_id, item_id_pos)
  df_ret$item_id_neg <- NA

  # sample from the non-positive items for each unique user id
  for (i in unique(df_ret$user_id)) {
    idx_user <- which(df_ret$user_id == i)
    df_user <- df_ret[idx_user, ]
    item_id_np <- setdiff(unique(df$item_id), df_user$item_id_pos)
    df_ret$item_id_neg[idx_user] <- sample(item_id_np, size = length(idx_user))
  }
  df_ret
}
