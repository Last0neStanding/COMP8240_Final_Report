import pandas as pd

data = pd.read_csv('product_info_amazon.csv')

print(data.columns)

# Fill NaN values in 'reviewText' with an empty string
data['Title'] = data['Title'].fillna('')

data['fasttext_format'] = '__label__' + data['Rating'].astype(str) + ' ' + data['Title']


# Save formatted data to a new file in FastText format
data[['fasttext_format']].to_csv('product_info_amazon_fasttext.txt', index=False, header=False)

print("Data converted successfully for FastText training.")

