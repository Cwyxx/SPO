import torch
from transformers import Dinov2Model

class Dinov2(torch.nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super(Dinov2, self).__init__()
        self.vision_model = Dinov2Model.from_pretrained(model_name)
        self.mlp = torch.nn.Linear(self.vision_model.config.hidden_size * 2, 1)
        
        # Freeze backbone
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
    
    def forward(self, x):
        outputs = self.vision_model(x)
        sequence_output = outputs[0] # batch_size, sequence_length, hidden_size
        
        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        logits = self.mlp(linear_input)
        
        return logits
        