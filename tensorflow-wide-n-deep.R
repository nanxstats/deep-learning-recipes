library("msaenet")
library("tensorflow")
library("tfestimators")

# load tensorflow-gpu in virtualenv folder
use_virtualenv("~/tensorflow/venv/")

# generate synthetic data for binary classification
# 1 million observations x 100 features (20 useful)
n <- 1e+6L
p <- 1e+2L
sim <- msaenet.sim.binomial(n = n, p = p, snr = 1, coef = rep(1, 20), p.train = 0.5)

# create input data: features
df_tr <- as.data.frame(sim$x.tr)
df_te <- as.data.frame(sim$x.te)

# set feature type so tensorflow recognizes them
feat <- vector("list", ncol(df_tr))
for (i in 1L:length(feat)) feat[[i]] <- column_numeric(paste0("V", i))

wide_columns <- feature_columns(feat)
deep_columns <- feature_columns(feat)

# define the "wide and deep" model
model <- dnn_linear_combined_classifier(
  linear_feature_columns = wide_columns,
  linear_optimizer = "Ftrl",
  dnn_feature_columns = deep_columns,
  dnn_optimizer = "Adam",
  dnn_hidden_units = c(50, 20, 10),
  dnn_dropout = 0.5
)

# add response to the input data
df_tr$y <- sim$y.tr
df_te$y <- sim$y.te

# input data constructor function
constructed_input_fn <- function(dataset)
  input_fn(dataset, features = -y, response = y)

train_input_fn <- constructed_input_fn(df_tr)
eval_input_fn <- constructed_input_fn(df_te)

# train the model
train(model, input_fn = train_input_fn, steps = 1e+4)
# 2018-08-19 20:50:13.846538: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
# 2018-08-19 20:50:13.846567: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2018-08-19 20:50:13.846572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0
# 2018-08-19 20:50:13.846575: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N
# 2018-08-19 20:50:13.846661: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9492 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
# Training 3907/10000 [============>....................] - ETA:  2m - loss: 12.08
# Training completed after 3907 steps but 10000 steps was specified

# evaluate model on the test set
metrics <- evaluate(model, input_fn = eval_input_fn, steps = 1e+4)
str(metrics)
# Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	1 obs. of  11 variables:
# $ accuracy            : num 0.745
# $ accuracy_baseline   : num 0.501
# $ auc                 : num 0.828
# $ auc_precision_recall: num 0.828
# $ average_loss        : num 0.507
# $ label/mean          : num 0.501
# $ loss                : num 64.9
# $ precision           : num 0.744
# $ prediction/mean     : num 0.502
# $ recall              : num 0.748
# $ global_step         : num 3907
