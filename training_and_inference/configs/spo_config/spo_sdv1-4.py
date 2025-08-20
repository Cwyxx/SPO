from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ###### Preference Model ######
    # config.preference_model_func_cfg = dict(
    #     type="hpsv2_preference_model_func"
    # )
    aigi_detector_path_dict = {
        "fatformer" : "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/FatFormer/fatformer_4class_ckpt.pth",
        "dinov2": "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/dinov2/genimage/best_model/model.safetensors",
        "dinov2-full_train": "/data_center/data2/dataset/chenwy/21164-data/model-ckpt/dinov2-full_train/genimage/best_model/model.safetensors",
        "drct_convb-sdv14": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/DRCT/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth",
        "drct_clip-sdv14": "/data_center/data2/dataset/chenwy/21164-data/detection-method-ckpt/DRCT/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth",
    }
    aigi_detector = "drct_clip-sdv14"
    config.preference_model_func_cfg = dict(
        type="aigi_detector_preference_model_func",
        aigi_detector=f"{aigi_detector}",
        aigi_detector_path=aigi_detector_path_dict[aigi_detector]
    )
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1
    config.num_epochs = 2
    
    #### logging ####
    # total_train_batch_size = 4 * 1 * 4 = 16
    config.train.early_stop_threshold = None
    config.train.save_and_eval_batch_interval = 250
    config.logdir = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/spo"
    config.run_name = f"{aigi_detector}"# config.run_name = "spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10"
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1

    return config
