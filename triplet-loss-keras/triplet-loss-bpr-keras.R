# Matrix factorization with BPR triplet loss and Keras -------------------------

library("keras")
library("magrittr")
library("reshape2")
library("ggplot2")
library("gridExtra")
library("hrbrthemes")
library("ggsci")

use_virtualenv("~/tensorflow/venv/")

# 1. Define BPR triplet loss ---------------------------------------------------

# identity loss: workaround Keras loss definition to use custom triplet loss
# there is no true label: we just want to minimize the BPR triplet loss
# to learn the embeddings
loss_identity <- function(y_true, y_pred) k_mean(y_pred - 0 * y_true)

# BPR triplet loss
loss_bpr_triplet <- function(x) {
  embed_user <- x[[1]]
  embed_item_positive <- x[[2]]
  embed_item_negative <- x[[3]]

  loss <- 1.0 - k_sigmoid(
    k_sum(embed_user * embed_item_positive, axis = -1, keepdims = TRUE) -
      k_sum(embed_user * embed_item_negative, axis = -1, keepdims = TRUE)
  )

  loss
}

# build and compile the model with BPR triplet loss
build_model <- function(n_user, n_item, n_factor) {

  # input layer for users
  input_user <- layer_input(shape = c(1), name = "input_user")

  # input layers for items (positive and negative)
  input_item_positive <- layer_input(shape = c(1), name = "input_item_positive")
  input_item_negative <- layer_input(shape = c(1), name = "input_item_negative")

  # embedding layer for users
  embed_user <- input_user %>%
    layer_embedding(
      input_dim = n_user, output_dim = n_factor,
      input_length = 1, name = "embed_user"
    ) %>%
    layer_flatten()

  # embedding layer shared by positive and negative items
  layer_embed_item <- layer_embedding(
    input_dim = n_item, output_dim = n_factor,
    input_length = 1, name = "embed_item"
  )

  embed_item_positive <- input_item_positive %>%
    layer_embed_item() %>%
    layer_flatten()
  embed_item_negative <- input_item_negative %>%
    layer_embed_item() %>%
    layer_flatten()

  # BPR triplet loss is the output
  loss <- list(embed_user, embed_item_positive, embed_item_negative) %>%
    layer_lambda(loss_bpr_triplet, output_shape = c(1))

  # define model inputs/outputs
  model <- keras_model(
    inputs = c(input_user, input_item_positive, input_item_negative),
    outputs = loss
  )

  # compile model
  model %>% compile(loss = loss_identity, optimizer = optimizer_nadam())

  model
}

# 2. Prepare the data ----------------------------------------------------------

# set model parameters

k <- 100 # number of latent factors to learn
n_epochs <- 20 # number of epochs

# read data
source("data-movielens.R")
movielens <- get_movielens_data()

# prepare training and test data
data_train <- movielens$train
data_test <- movielens$test

# model constants: remember to include all ids from train and test
n_user <- length(unique(c(data_train$user_id, data_test$user_id))) + 1L
n_item <- length(unique(c(data_train$item_id, data_test$item_id))) + 1L

# prepare the test triplets
triplets_test <- get_triplets(data_test)

# 3. Train the model -----------------------------------------------------------

# build the model
model <- build_model(n_user, n_item, k)

# inspect the model
summary(model)

# sanity check: untrained model's auc should be around 0.5
source("metric-auc.R")
auc_avg(model, data_test)

# training loop
train_loss <- test_loss <- train_auc <- test_auc <- rep(NA, n_epochs)

for (epoch in 1L:n_epochs) {
  cat("Epoch", epoch, "\n")

  # sample triplets from the training data
  triplets_train <- get_triplets(data_train)

  history_train <- model %>%
    fit(
      x = list(
        "input_user" = matrix(triplets_train$user_id, ncol = 1),
        "input_item_positive" = matrix(triplets_train$item_id_pos, ncol = 1),
        "input_item_negative" = matrix(triplets_train$item_id_neg, ncol = 1)
      ),
      y = matrix(1, nrow = length(triplets_train$user_id), ncol = 1),
      batch_size = 64, epochs = 1, verbose = 1, shuffle = TRUE
    )

  train_loss[epoch] <- history_train$metrics$loss
  train_auc[epoch] <- auc_avg(model, data_train)
  cat("AUC train:", train_auc[epoch], "\n")

  history_test <- model %>%
    evaluate(
      x = list(
        "input_user" = matrix(triplets_test$user_id, ncol = 1),
        "input_item_positive" = matrix(triplets_test$item_id_pos, ncol = 1),
        "input_item_negative" = matrix(triplets_test$item_id_neg, ncol = 1)
      ),
      y = matrix(1.0, nrow = length(triplets_test$user_id), ncol = 1),
      batch_size = 64, verbose = 0
    )

  test_loss[epoch] <- unname(history_test)
  test_auc[epoch] <- auc_avg(model, data_test)
  cat("AUC test:", test_auc[epoch], "\n")
}

# 4. Plot loss and user-averaged AUC -------------------------------------------

df_loss <-
  data.frame("epoch" = 1L:n_epochs, "train" = train_loss, "test" = test_loss)
df_loss <-
  melt(df_loss, id.vars = "epoch", variable.name = "data", value.name = "loss")
p_loss <- ggplot(df_loss, aes(x = epoch, y = loss)) +
  geom_point(aes(fill = data), shape = 21, colour = "#333333") +
  geom_smooth(aes(colour = data), span = 0.5, se = FALSE, show.legend = FALSE) +
  theme_ipsum() +
  theme(plot.margin = unit(c(1, 1, 0, 1), "cm")) +
  scale_fill_tron() +
  scale_color_tron()

df_auc <-
  data.frame("epoch" = 1L:n_epochs, "train" = train_auc, "test" = test_auc)
df_auc <-
  melt(df_auc, id.vars = "epoch", variable.name = "data", value.name = "auc")
p_auc <- ggplot(df_auc, aes(x = epoch, y = auc)) +
  geom_point(aes(fill = data), shape = 21, colour = "#333333") +
  geom_smooth(aes(colour = data), span = 0.5, se = FALSE, show.legend = FALSE) +
  theme_ipsum() +
  theme(plot.margin = unit(c(0, 1, 1, 1), "cm")) +
  scale_fill_tron() +
  scale_color_tron()

p <- grid.arrange(p_loss, p_auc, nrow = 2)

ggsave("triplet-loss-bpr-movielens.png", p, width = 9, height = 6)
