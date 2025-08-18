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
    config.bool_spo_reward_aigi_detector_func = True
    config.preference_model = "hpsv2"
    config.aigi_detector = "dinov2"
    config.aigi_detector_weight = 0.1
    config.preference_model_func_cfg = dict(
        type="spo_reward_aigi_detector_func",
        reward_model_func_cfg=dict(
            type=f"{config.preference_model}_preference_model_func",
        ),
        aigi_detector_func_cfg=dict(
            type=f"aigi_detector_preference_model_func", 
            aigi_detector=f"{config.aigi_detector}",
            aigi_detector_path=f"/data_center/data2/dataset/chenwy/21164-data/model-ckpt/{config.aigi_detector}/genimage/best_model/model.safetensors"
        )
    )
    
    ###### Compare Function ######
    compare_func_threshold = 1e-2
    config.compare_func_cfg = dict(
        type="spo_reward_aigi_detector_compare",
        threshold=compare_func_threshold,
        aigi_detector_weight=config.aigi_detector_weight
    )
    
    
    ###### Training ######
    config.train.early_stop_threshold = 0.3
    config.sample.num_sample_each_step = 4
    # total_train_batch_size = 4 * 1 * 4 = 16
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1
    config.resume_from = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/spo/spo-hpsv2_0.9-dinov2_0.1-comp_func_threshold_0.01/checkpoint_0"
    config.num_epochs = 4
    
    #### logging ####
    config.wandb_project_name = "spo"
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/{config.wandb_project_name}"
    config.run_name = f"{config.wandb_project_name}-{config.preference_model}_{1-config.aigi_detector_weight}-{config.aigi_detector}_{config.aigi_detector_weight}-comp_func_threshold_{compare_func_threshold}"
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1
    
    return config
