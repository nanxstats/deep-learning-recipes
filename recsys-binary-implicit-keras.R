library("keras")
library("magrittr")
library("progress")
library("hrbrthemes")
library("ggsci")

use_virtualenv("~/tensorflow/venv/")

# 1. preprocess data -----------------------------------------------------------

# see full dataset at https://github.com/nanxstats/MEF/
url_drug <- "https://raw.githubusercontent.com/nanxstats/MEF/master/data/drug.txt"
url_adr <- "https://raw.githubusercontent.com/nanxstats/MEF/master/data/adr.txt"
url_pair <- "https://raw.githubusercontent.com/nanxstats/MEF/master/data/association.txt"

df_drug <- read.table(url_drug, sep = "\t", as.is = TRUE)
df_adr <- read.table(url_adr, sep = "\t", as.is = TRUE)
df_pair <- read.table(url_pair, sep = "\t", as.is = TRUE)

names(df_drug) <- c("drug_id", "drugbank_id", "kegg_id", "name", "smiles")
names(df_adr) <- c("adr_id", "name")
names(df_pair) <- c("drug_id", "adr_id")

# clean up drug id
clean_drug_id <- function(x) gsub("-", "", as.character(x))
df_drug$drug_id <- clean_drug_id(df_drug$drug_id)
df_pair$drug_id <- clean_drug_id(df_pair$drug_id)

# map drug id and adr id to integer numbers (for the input)
for (i in 1L:nrow(df_pair)) {
  df_pair$"drug_id"[i] <- which(df_pair$"drug_id"[i] == df_drug$"drug_id") - 1L
  df_pair$"adr_id"[i] <- which(df_pair$"adr_id"[i] == df_adr$"adr_id") - 1L
}

# complete the unobserved rows as class 0
pair <- expand.grid(
  "drug_id" = unique(df_pair$drug_id),
  "adr_id" = unique(df_pair$adr_id),
  stringsAsFactors = FALSE
)
pair$"class" <- 0L

# set known drug-ADR association pairs as class 1
pb <- progress_bar$new(
  format = "[:bar] :percent eta: :eta", total = nrow(df_pair)
)
for (i in 1L:nrow(df_pair)) {
  pb$tick()
  pair[intersect(
    which(df_pair$drug_id[i] == pair$drug_id),
    which(df_pair$adr[i] == pair$adr_id)
  ), "class"] <- 1L
}

# convert to integer
pair$drug_id <- as.integer(pair$drug_id)
pair$adr_id <- as.integer(pair$adr_id)

# shuffle rows
set.seed(42)
pair <- pair[sample(1L:nrow(pair)), ]

# 2. matrix factorization with Keras -------------------------------------------

# basic settings
n_drug <- length(unique(df_pair$drug_id))
n_adr <- length(unique(df_pair$adr_id))
k <- 10 # number of latent factors to learn

# input layers
input_drug <- layer_input(shape = c(1))
input_adr <- layer_input(shape = c(1))

# embedding and flatten layers
embed_drug <- input_drug %>%
  layer_embedding(input_dim = n_drug, output_dim = k, input_length = 1) %>%
  layer_flatten()
embed_adr <- input_adr %>%
  layer_embedding(input_dim = n_adr, output_dim = k, input_length = 1) %>%
  layer_flatten()

# dot product and output layer (can be replaced by arbitrary DNN architecture)
pred <- layer_dot(list(embed_drug, embed_adr), axes = -1) %>%
  layer_dense(units = 1, activation = "sigmoid")

# define model inputs/outputs
model <- keras_model(inputs = c(input_drug, input_adr), outputs = pred)
model %>% compile(
  loss = "binary_crossentropy",
  metric = "binary_accuracy",
  optimizer = optimizer_rmsprop() # the most stable one here
)

# inspect model
summary(model)

# train the model
history <- model %>% fit(
  x = list(
    matrix(pair$drug_id, ncol = 1),
    matrix(pair$adr_id, ncol = 1)
  ),
  y = matrix(pair$class, ncol = 1),
  class_weight = list("1" = 50.0, "0" = 1.0), # deal with unbalanced classes
  epochs = 20,
  batch_size = 2000, # needs some tuning
  validation_split = 0.2
)

# plot training history
plot(history) +
  theme_ipsum() +
  scale_color_startrek() +
  scale_fill_startrek()
