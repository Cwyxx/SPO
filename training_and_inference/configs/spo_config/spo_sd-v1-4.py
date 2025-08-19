from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ###### Preference Model ######
    # config.preference_model_func_cfg = dict(
    #     type="hpsv2_preference_model_func"
    # )
    aigi_detector = "dinov2"
    config.preference_model_func_cfg = dict(
        type="aigi_detector_preference_model_func",
        aigi_detector=f"{aigi_detector}",
        aigi_detector_path=f"/data_center/data2/dataset/chenwy/21164-data/model-ckpt/{aigi_detector}/genimage/best_model/model.safetensors"
    )
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1
    config.num_epochs = 4
    
    #### logging ####
    # total_train_batch_size = 4 * 1 * 4 = 16
    config.train.early_stop_threshold = 0.4
    config.train.early_stop_warmup_step = 4000
    config.logdir = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/spo"
    config.run_name = f"{aigi_detector}"# config.run_name = "spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10"
    config.resume_from = None
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1

    return config
