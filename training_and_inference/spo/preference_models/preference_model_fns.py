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
    _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        img = _transform(img)
        caption = hpsv2.tokenizer(extra_info["prompts"]).to(cfg.device)
        outputs = hpsv2.model(img, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits) # reward
        return scores

    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="imagereward_preference_model_func")
def imagereward_preference_model_func_builder(cfg):
    import ImageReward as RM
    imagereward = RM.load("ImageReward-v1.0", device=cfg.device).eval().requires_grad_(False)
    _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        img = _transform(img)
        
        rm_input_ids, rm_attention_masks = [], []
        for prompt in extra_info['prompts']:
            tokenizer_dict = imagereward.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
            rm_input_ids.append(tokenizer_dict.input_ids)
            rm_attention_masks.append(tokenizer_dict.attention_mask)
            
        rm_input_ids = torch.stack(rm_input_ids)
        rm_attention_masks = torch.stack(rm_attention_masks)
        rm_input_ids = rm_input_ids.view(-1, rm_input_ids.shape[-1]).to(cfg.device)
        rm_attention_masks = rm_attention_masks.view(-1, rm_attention_masks.shape[-1]).to(cfg.device)
        
        scores = imagereward.score_gard(rm_input_ids, rm_attention_masks, img) # reward
        return scores
    
    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="pickscore_preference_model_func")
def pickscore_preference_model_func_builder(cfg):
    from transformers import AutoProcessor, AutoModel
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path)
    pickscore = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(cfg.device).requires_grad_(False)
    _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        img = _transform(img)
        
        text_inputs = pickscore_processor(text=extra_info["prompts"], padding=True, truncation=True, max_length=77, return_tensors="pt").to(cfg.device)
        image_embs = pickscore.get_image_features(img)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True) # [B, embedding_dim]
        text_embs = pickscore.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True) # [B, embedding_dim]
    
        logits = pickscore.logit_scale.exp() * text_embs @ image_embs.T # [B, B]
        scores = torch.diagonal(logits) # [B]
        
        return scores
    
    return preference_fn

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name="aigi_detector_preference_model_func")
def aigi_detector_preference_model_func_builder(cfg):
    import safetensors
    if cfg.aigi_detector == "univfd":
        from aigi_detector_training.univfd import UnivFD
        aigi_detector = UnivFD("openai/clip-vit-large-patch14")
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif cfg.aigi_detector == "dinov2":
        from aigi_detector_training.dinov2 import Dinov2
        aigi_detector = Dinov2("facebook/dinov2-base")
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    ckpt = safetensors.torch.load_file(cfg.aigi_detector_path)
    aigi_detector.load_state_dict(ckpt)
    aigi_detector.eval().to(cfg.device).requires_grad_(False)
    
    def preference_fn(img, extra_info):
        img = (img / 2 + 0.5).clamp(0, 1).float()
        img = _transform(img)

        logits = aigi_detector(img)
        outputs = torch.sigmoid(logits) # 0 -> real, 1 -> fake
        scores = 1 - outputs
        
        return scores
    
    return preference_fn