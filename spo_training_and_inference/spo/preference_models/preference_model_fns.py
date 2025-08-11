import torch
import torchvision

from .builder import PREFERENCE_MODEL_FUNC_BUILDERS

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='step_aware_preference_model_func')
def step_aware_preference_model_func_builder(cfg):
    from .models.step_aware_preference_model import StepAwarePreferenceModel
    step_aware_preference_model = StepAwarePreferenceModel(
        model_pretrained_model_name_or_path=cfg.model_pretrained_model_name_or_path,
        processor_pretrained_model_name_or_path=cfg.processor_pretrained_model_name_or_path,
        ckpt_path=cfg.ckpt_path,
    ).eval().to(cfg.device)
    step_aware_preference_model.requires_grad_(False)
    
    @torch.no_grad()
    def preference_fn(img, extra_info):
        # b
        scores = step_aware_preference_model.get_preference_score(
            img, 
            extra_info['input_ids'],
            extra_info['timesteps'],
        )
        return scores
    
    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='hpsv2_preference_model_func')
def hpsv2_preference_model_func_builder(cfg):
    from .models.hpsv2 import HPSv2
    hpsv2 = HPSv2()
    hpsv2.eval().to(cfg.device)
    hpsv2.requires_grad_(False)
    
    @torch.no_grad()
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
            
        img = _transform(img)
        caption = hpsv2.tokenizer(extra_info["prompts"]).to(cfg.device)
        outputs = hpsv2.model(img, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        return scores

    return preference_fn

        