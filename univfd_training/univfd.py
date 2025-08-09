import torch
from transformers import CLIPVisionModel

class UnivFD(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super(UnivFD, self).__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.mlp = torch.nn.Linear(self.vision_model.config.hidden_size, 1)
        
        # Freeze backbone
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        pooler_output = self.vision_model(x).pooler_output # pooler_output, [batch_size, hidden_size], [CLS] token -> LayerNorm
        logits = self.mlp(pooler_output)
        return logits