from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ###### Preference Model ######
    config.preference_model_func_cfg = dict(
        type="hpsv2_preference_model_func"
    )
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    config.train.train_batch_size = 4
    config.train.gradient_accumulation_steps = 1
    
    #### logging ####
    # total_train_batch_size = 4 * 1 * 4 = 16
    config.logdir = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/SPO"
    config.run_name = "hpsv2-spo_sd-v1-4_4k-prompts_num-sam-4_10ep_bs1"# config.run_name = "spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10"
    config.resume_from = "/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k/SPO/hpsv2-spo_sd-v1-4_4k-prompts_num-sam-4_10ep_bs1/checkpoint_5"

    return config
