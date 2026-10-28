############################# SwinT_model_modules ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: creating SwinT model by loading pretrained weight for searching the best learning rate

########################## API Section #########################
from SOTAs_main_func.SIL_Model.Mixer_models.models.mixer import MLPMixer
from torch import nn
import torch
from torchsummaryX import summary

def creating_Mixer(gpu_device=0, class_num=3):
    mlp_mixer_base = MLPMixer(image_size=(224, 224), channels=3, patch_size=16, dim=768, depth=12, num_classes=3,
                              expansion_factor=4, expansion_factor_token=0.5, dropout=0.)
    #mixer_weight_file = torch.load(r'E:\SOTA_Model_Classification\SC_Weights'
                                   #r'\Mixer\mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth')
    #mixer_weight_file = {k: v for k, v in mixer_weight_file.items() if (k in mixer_weight_file and 'meta' not in k)}
    # vit_weight_file = {k: v for k, v in vit_weight_file.items() if
    #                   (k in vit_weight_file and 'patch_embed.proj' not in k)}
    #mlp_mixer_base.load_state_dict(mixer_weight_file, strict=False)
    print(mlp_mixer_base[12][1].fn[0].weight)

    #with torch.no_grad():
    #    print('########################## mlp_mixer_summary #########################')
    #    summary(mlp_mixer_base, torch.randn((1, 3, 224, 224)))
    #    print('\n', '########################## mlp_mixer_net #########################')
    #    print(mlp_mixer_base, '\n')

    mlp_mixer_base = mlp_mixer_base.cuda(gpu_device)
    return mlp_mixer_base