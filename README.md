[mnist demo.webm](https://github.com/vishalsmb/vector-image-search-mnist/assets/30661709/fadd38fc-db5d-42ad-9da7-48cd890ab3e8)# vector-image-search-mnist
This code repo illustrates the power of image search using vector databases and embeddings.
The steps involved are as follows.
1. Download the mnist dataset
2. Train and save an autoencoder model
3. Take the encoder head and create embeddings on the train split
4. Index them to a vector database index 
5. Create a streamlit app
6. Load the saved model
7. Accept an input image from the user 
8. Convert it to embedding
9. Query the vector database
10. Display the results

## Steps to run locally
1. Python version used : 3.10
2. Install all dependencies using ```pip install -r requirements.txt```
3. If you want to train your own autoencoder model else skip this step to use the pretrained model in the repo.
   1. Run ```pytorch_autoencoder.py``` to train and save your autoencoder model
4. Indexing in Pinecone
   1. Create an index in Pinecone
   2. Set your API key and index as env variables [```PINECONE_API_KEY```, ```PINECONE_INDEX_NAME```]
   3. Uncomment lines #57 and #58 in ```pinecone_utils.py``` and run it to index the data to the vector db
   4. Upon successful indexing, you now have your custom trained embeddings indexed.
6. Now run the streamlit app with the below command :
   1. ```streamlit run index.py``` 
7. You can select any mnist from your local directory and the app would fetch similar images 

The intent of this repo is to illustrate the capability of searching similar images by generating 
custom embeddings and indexing it to a vector database. Though its not necessary to use solutions like vector 
databases for smaller search spaces (like in the case of MNIST, 60K images), this approach can be used as template
to index and search efficiently among millions of images.

Streamlit app URL : https://vector-image-search-mnist-kk1rhwtgl8.streamlit.app/

Usage: Input any mnist image and similar ones would be fetched

[mnist demo.webm](https://github.com/vishalsmb/vector-image-search-mnist/assets/30661709/7277da7f-46b8-448b-8b25-f2eeff6702f4)
