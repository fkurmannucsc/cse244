import torch

def create_same_image_mask(labels):
    return torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()

def cosine_similarity(embedding, temperature):
    normalized_embedding = torch.nn.functional.normalize(embedding, dim=1)
    return torch.matmul(normalized_embedding, normalized_embedding.T) / temperature
