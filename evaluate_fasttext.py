import fasttext

# load saved model
model = fasttext.train_supervised(
    input="product_info_amazon_fasttext.txt",
    epoch=50,         
    lr=0.7,           
    wordNgrams=2,
    dim=100       
)


# evaluation by using fasttext test()
result = model.test("product_info_amazon_fasttext.txt")

# get result
print("Number of examples:", result[0])  
print("Precision:", result[1])           
print("Recall:", result[2])              


