from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ##### dataset #####
    config.dataset_cfg = dict(
        type="PromptDataset",
        meta_json_path='assets/prompts/spo_4k.json',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    ###### Preference Model ######
    preference_model = "hpsv2"
    aigi_detector = "dinov2"
    aigi_detector_weight = 0.5
    config.preference_model_func_cfg = dict(
        type="spo_reward_aigi_detector_func",
        reward_model_func_cfg=dict(
            type=f"{preference_model}_preference_model_func",
        ),
        aigi_detector_func_cfg=dict(
            type=f"aigi_detector_preference_model_bceloss_func", 
            aigi_detector=f"{aigi_detector}",
            aigi_detector_path=f"/data_center/data2/dataset/chenwy/21164-data/model-ckpt/{aigi_detector}/genimage/best_model/model.safetensors"
        )
    )
    
    ###### Compare Function ######
    config.compare_func_cfg = dict(
        type="spo_reward_aigi_detector_compare",
        threshold=0.0,
        aigi_detector_weight=config.aigi_detector_weight
    )
    
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    # total_train_batch_size = 4 * 1 * 4 = 16
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1
    config.resume_from = None
    config.num_epochs = 4
    
    #### logging ####
    config.wandb_project_name = "spo"
    config.logdir = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/SPO"
    config.run_name = f"{config.wandb_project_name}-{preference_model}_{1-config.aigi_detector_weight}-{aigi_detector}_{config.aigi_detector_weight}"

    return config
