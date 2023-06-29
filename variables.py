import pinecone
import os
import torch
from model_definition import AutoEncoder
import logging

# Logging format
log_format = "%(asctime)s::%(levelname)s::%(name)s::" \
             "%(filename)s::%(lineno)d::%(message)s"

# Pinecone init
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment='gcp-starter')
pinecone_index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the encoder model
model = AutoEncoder()
model.load_state_dict(torch.load("autoencoder.pt"))
model.to(device)
model.eval()
