from transformers import AutoModel
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Initialize the model
model = AutoModel.from_pretrained("/data/cys-hx/MMR-LR1/checkpoints/jinaai/jina-embeddings-v4", trust_remote_code=True, torch_dtype=torch.float16)

model.to("cuda")

# ========================
# 1. Retrieval Task
# ========================
# Configure truncate_dim, max_length (for texts), max_pixels (for images), vector_type, batch_size in the encode function if needed

# Encode query
query_embeddings = model.encode_text(
    texts=["Overview of climate change impacts on coastal cities"],
    task="retrieval",
    prompt_name="query",
)

print("Query Embeddings:", query_embeddings)

# # Encode passage (text)
# passage_embeddings = model.encode_text(
#     texts=[
#         "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
#     ],
#     task="retrieval",
#     prompt_name="passage",
# )

# # Encode image/document
# image_embeddings = model.encode_image(
#     images=["https://i.ibb.co/nQNGqL0/beach1.jpg"],
#     task="retrieval",
# )
