library("reticulate")

# write to png (low-res)
k <- import("keras")

plot_model <- k$utils$plot_model
plot_model(model, to_file = "triplet-loss-model-keras.png", show_shapes = TRUE, show_layer_names = TRUE)

# write to pdf
pydot <- import("pydot")

model_to_dot <- k$utils$vis_utils$model_to_dot
g <- model_to_dot(model)
g$write_pdf("triplet-loss-model-keras.pdf")
