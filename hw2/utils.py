import torch
import torch.nn as nn
from tqdm import tqdm


def cal_num_params(model):
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    return num_trainable_parameters

def find_feats(data_loader, model, device, prefix='dev/'):
    model.eval()

    # Find feature dict
    feats_dict = dict()
    for batch_idx, (imgs, path_names) in tqdm(enumerate(data_loader), total=len(data_loader), 
                                              position=0, leave=False, desc='Feat_dict'):
        imgs = imgs.to(device)

        # Return feature embedding
        with torch.no_grad():
            feats = model(imgs, return_feats=True) 
        
        # Update feature dict
        feats_dict.update({prefix + i: j for i, j in zip(path_names, feats)})
    
    return feats_dict


def cal_similarity(a, b, feats_dict):
    # Calculate similarity
    similarity_metric = nn.CosineSimilarity(dim=0)
    similarity = similarity_metric(feats_dict[a], feats_dict[b]).item()
    return similarity


# https://github.com/NVIDIA/DeepLearningExamples
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
