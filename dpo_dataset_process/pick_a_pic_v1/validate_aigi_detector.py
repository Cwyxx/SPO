import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # pick_a_pic_v1
preference_model_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "training_and_inference", "spo")) # preference_models
grandparent_dir = osp.abspath(osp.join(current_dir, "..", "..")) # aigi_detector
sys.path.insert(0, preference_model_dir)
import argparse
import torch
import torchvision


def get_aigi_detector_and_transform(cfg):
    import safetensors
    if aigi_detector == "univfd":
        from aigi_detector_training.univfd import UnivFD
        aigi_detector = UnivFD("openai/clip-vit-large-patch14")
        ckpt = safetensors.torch.load_file(cfg.aigi_detector_path)
        aigi_detector.load_state_dict(ckpt)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif cfg.aigi_detector.startswith("dinov2"):
        from aigi_detector_training.dinov2 import Dinov2
        aigi_detector = Dinov2("facebook/dinov2-base")
        ckpt = safetensors.torch.load_file(cfg.aigi_detector_path)
        aigi_detector.load_state_dict(ckpt)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    elif cfg.aigi_detector == "fatformer":
        from preference_models.fatformer_models import build_model as fatformer_build_model
        cfg.backbone = "CLIP:ViT-L/14"
        cfg.num_classes = 2
        cfg.num_vit_adapter = 8
        cfg.num_context_embedding = 8
        cfg.init_context_embedding = ""
        cfg.hidden_dim = 768
        cfg.clip_vision_width = 1024
        cfg.frequency_encoder_layer = 2
        cfg.decoder_layer = 4
        cfg.num_heads = 12
        aigi_detector = fatformer_build_model(cfg)
        ckpt = torch.load(cfg.aigi_detector_path, map_location='cpu')
        aigi_detector.load_state_dict(ckpt['model'], strict=False)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    elif cfg.aigi_detector in [ "drct_convb-sdv14", "drct_clip-sdv14", "drct_convb-sdv21", "drct_clip-sdv21"]:
        from preference_models.drct_models.models import get_models as drct_get_models
        model_name_dict = { "drct_convb-sdv14": "convnext_base_in22k", "drct_convb-sdv21": "convnext_base_in22k", "drct_clip-sdv14": "clip-ViT-L-14", "drct_clip-sdv21": "clip-ViT-L-14" }
        cfg.is_train = False
        cfg.num_classes = 2
        cfg.freeze_extractor = False
        cfg.embedding_size = 1024
        aigi_detector = drct_get_models(model_name=model_name_dict[cfg.aigi_detector], pretrained=cfg.is_train, num_classes=cfg.num_classes, freeze_extractor=cfg.freeze_extractor, embedding_size=cfg.embedding_size)
        aigi_detector.load_state_dict(torch.load(cfg.aigi_detector_path, map_location='cpu'), strict=False)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    elif cfg.aigi_detector == "code":
        from preference_models.code_models.models import VITContrastiveHF as CoDE_Model
        cfg.classification_type = "linear"
        aigi_detector = CoDE_Model(cfg.classification_type, cfg.aigi_detector_path)
        
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    aigi_detector.eval().to(cfg.device).requires_grad_(False)
    return aigi_detector, _transform

parser = argparse.ArgumentParser(description="")
parser.add_argument("--aigi_detector", type=str)
args = parser.parse_args()

#### Aigi Detector Path ####
aigi_detector_path_dict = {
    "fatformer" : "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/FatFormer/fatformer_4class_ckpt.pth",
    "dinov2": "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/dinov2/genimage/best_model/model.safetensors",
    "dinov2-full_train": "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/dinov2-full_train/genimage/best_model/model.safetensors",
    "drct_convb-sdv14": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/DRCT/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth",
    "drct_clip-sdv14": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/DRCT/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth",
    "code": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/CoDE"
}
cfg = {}
cfg.device = torch.device('cuda')
cfg.aigi_detector = args.aigi_detector
cfg.aigi_detector_path = aigi_detector_path_dict[cfg.aigi_detector]
aigi_detector, _transform = get_aigi_detector_and_transform(cfg)

base_pick_a_pic_dir = "/data_center/data2/dataset/chenwy/21164-data/dpo_dataset/pick_a_pic_v1"
pick_a_pic_id_list = os.listdir(base_pick_a_pic_dir)