import ml_collections
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
    preference_model = "imagereward"
    config.preference_model_func_cfg = dict(
        type=f"{preference_model}_preference_model_func"
    ) 
    
    ###### logging ######
    # total_batch_size: 1 * 4 * 4
    config.wandb_project_name = "drtune_cfg"
    config.logdir = f"/data_center/data2/dataset/chenwy/21164-data/stable_diffusion/stable_diffusion_v1_4/spo_4k"
    config.run_name = f"{config.wandb_project_name}-{preference_model}" # experiment name under a project (wandb_project_name) in swanlab.
    
    ###### Training ######
    config.resume_from = None
    config.num_epochs = 5
    config.pipeline_num_inference_steps = 50
    
    ###### drtune ######
    config.drtune = drtune = ml_collections.ConfigDict()
    drtune.K = 5 # number of training time steps (denoisign time steps)
    drtune.T = 50 # Total number of denoising steps used during finetuning.
    drtune.M = 20 # Maximal early stop step m.
    
    #### validation ####
    config.validation_prompts = [ 'a cat.', 'a dog', 'a horse.', 'A bus stopped on the side of the road while people board it.', 'A woman holding a plate of cake in her hand.']
    config.num_validation_images = 1

    return config