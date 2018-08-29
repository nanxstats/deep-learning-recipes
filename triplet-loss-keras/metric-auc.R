library("pROC")

# predict outcome probability
predict_prob <- function(model, user_id, item_id) {
  user_matrix <- model %>% get_layer("embed_user") %>% get_weights() %>% extract2(1L)
  user_vector <- user_matrix[user_id + 1L, , drop = FALSE]
  item_matrix <- model %>% get_layer("embed_item") %>% get_weights() %>% extract2(1L)
  item_vectors <- item_matrix[item_id + 1L, , drop = FALSE]

  sigmoid <- function(x) 1 / (1 + exp(-x))
  scores <- user_vector %*% t(item_vectors) %>% sigmoid() %>% as.vector()
  scores
}

# compute an "average AUC" for predictions
auc_avg <- function(model, df) {

  # all user ids in the set to evaluate on
  user_id <- unique(df$user_id)
  n_users <- length(user_id)

  # all possible items in the test set
  n_items <- max(df$item_id)
  item_id <- 1L:n_items

  # for each user, we make predictions on all possible items,
  # compute the auc, until all auc values for all # of users are computed.
  # then average all the auc values.
  scores <- rep(NA, n_users)

  for (i in 1L:n_users) {
    prob <- predict_prob(model, user_id[i], item_id)
    label <- rep(0L, n_items)
    # fill in positive labels
    item_id_pos <- df$item_id[which(df$user_id == user_id[i] & df$rating == 1L)]
    if (length(item_id_pos) >= 1L) {
      label[item_id_pos] <- 1L
      # the argument `direction` is super important here...
      # since unlike sklearn.metrics.roc_auc_score,
      # pROC will always report the auc > 0.5 when direction = "auto".
      # so the mean value of all these aucs will be significantly > 0.5.
      scores[i] <- as.numeric(auc(label, prob, direction = "<"))
    } else {
      # all zeros exception handling: AUC requires the label to have two levels
      scores[i] <- NA
    }
  }

  mean(scores, na.rm = TRUE)
}
