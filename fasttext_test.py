import fasttext

# training model
model = fasttext.train_supervised(input="product_info_amazon_fasttext.txt")

# save model
model.save_model("product_info_amazon_model.bin")

print("Model trained and saved successfully!")
