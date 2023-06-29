import shutil
import pinecone
import os
from tqdm import tqdm
import torchvision.datasets as datasets
import uuid
from general_utils import download_mnist_dataset, preprocess
import numpy as np
import torch
from PIL import Image
import logging
from variables import device, log_format

logging.basicConfig(level='INFO', format=log_format)


def insert_into_index(index, embeddings, batch):
    request_list = [{"id": str(uuid.uuid4()),
                     "values": embedding,
                     "metadata": {'file_path': batch[i]}
                     } for i, embedding in enumerate(embeddings)]

    return index.upsert(
        vectors=request_list
    )


def convert_to_embedding(encoder, image_paths):
    images = preprocess(image_paths)
    output = encoder(images)
    return output.cpu().detach().numpy().tolist()


def index_to_pinecone(model, pinecone_index):
    train_dir, test_dir = download_mnist_dataset()
    files = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    batches = [files[i:i+100] for i in range(0, len(files), 100)]
    for batch in tqdm(batches):
        try:
            embeddings = convert_to_embedding(model.encoder, batch)
            response = insert_into_index(pinecone_index, embeddings, batch)
        except BaseException as bx:
            logging.error("Exception during indexing : {}".format(bx))
            raise bx


def query_index(index, query, top_k=5):
    return index.query(
        top_k=5,
        include_values=False,
        include_metadata=True,
        vector=query
    )

# uncomment these two lines to index embeddings to pinecone
# from variables import model, pinecone_index
# index_to_pinecone(model, pinecone_index)