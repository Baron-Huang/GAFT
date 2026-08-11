import torch
import io
import contextlib
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import random
from utils.train_utils import train_single, train_bw_EMCO, train_moo, testing_funnction, CapturePrint, PrintCapture
from models.swin_transformer import swin_tiny
from models.gcn import vig_ti_224_gelu
from models.my_net import ViT2GNN_w


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    print(f"Random seed set as {seed}")


def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)
    torch.cuda.manual_seed(7 + worker_id)
    torch.cuda.manual_seed_all(7 + worker_id)

if __name__ == '__main__':
    setup_seed(0)
    class_num = 3
    batch_size = 32
    epochs = 100
    image_size = 224
    num_workers = 1
    data_root = r'./dataset/esophageal_cancer'
    swinT_path = r'./weight/swin_tiny_patch4_window7_224_22k.pth'
    gcn_path = r'./weight/vig_ti_74.5.pth'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('########################## reading datas and processing datas #########################')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size], antialias=True),
                                    transforms.Normalize(mean=0.5, std=0.5)])
    train_data = ImageFolder(data_root + r'/Train_patch', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = ImageFolder(data_root + r'/Val_patch', transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_data = ImageFolder(data_root + r'/Test_patch', transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for i in range(8):
        ###########################################Swin_Part################################################
        swin_base = swin_tiny(class_num=class_num)
        weight_file = torch.load(swinT_path, map_location='cuda:1')["model"]
        weight_file = {k: v for k, v in weight_file.items() if (k in weight_file and 'head' not in k)}
        swin_base.load_state_dict(weight_file, strict=False)
        nn.init.trunc_normal_(swin_base.head.weight, std=.02)
        ###########################################GCN_Part################################################
        gcn_base = vig_ti_224_gelu(class_num=class_num) #gat
        weight_file = torch.load(gcn_path, map_location='cuda:1')
        weight_file = {k: v for k, v in weight_file.items() if (k in weight_file and 'prediction.4' not in k)}
        gcn_base.load_state_dict(weight_file, strict=False)
        nn.init.trunc_normal_(gcn_base.prediction[4].weight, std=.02)
        with torch.no_grad():
            fusion_net = ViT2GNN_w(swinT_base=swin_base, vig_base=gcn_base)
            #print('\n', '########################## Net_summary #########################')
            #summary(model=m_net, input_size=(3,224,224), device='cuda')
            #print(fusion_net)
            # print(fusion_net.prediction[0].weight)
            # print(fusion_net.head.weight)
            # print(fusion_net.vig_out.weight)
        fusion_net = fusion_net.to(device)
        fusion_weight = 0.4
        probability_weight = 1.0 + 0.1 * i
        entropy_weight = 1.0
        weight_save_path = f'./result/esophagel/{fusion_weight}*fusion_weight+{probability_weight}*probability_weight+{entropy_weight}*entropy_weight.pth'
        log_file_path = f'./result/esophagel/{fusion_weight}*fusion_weight+{probability_weight}*probability_weight+{entropy_weight}*entropy_weight.txt'

        capture_1 = CapturePrint()
        capture_2 = CapturePrint()
        print_capture_1 = PrintCapture(capture_1)
        print_capture_2 = PrintCapture(capture_2)

        with contextlib.redirect_stdout(print_capture_1) as f:
            train_bw_EMCO(number=i+1, fusion_net=fusion_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                            gpu_device=device, epoch=epochs, weight_path=weight_save_path,fusion_weight=fusion_weight,probability_weight=probability_weight,entropy_weight=entropy_weight)
            train_logs = f.getvalue()
        with open(log_file_path, 'w') as f:
            f.write('training logs:\n')
            f.write(train_logs)

        with contextlib.redirect_stdout(print_capture_2) as f:
            testing_funnction(test_model=fusion_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                              gpu_device=device, out_mode='five')
            test_logs = f.getvalue()
        with open(log_file_path, 'a') as f:
            f.write('\ntesting logs:\n')
            f.write(test_logs)



    """###########################################GCN_Part################################################
    gcn_net = vig_ti_224_gelu(class_num=class_num)
    weight_file = torch.load(opt.gcn_path)
    weight_file = {k: v for k, v in weight_file.items() if (k in weight_file and 'prediction.4' not in k)}
    gcn_net.load_state_dict(weight_file, strict=False)
    nn.init.trunc_normal_(gcn_net.prediction[4].weight, std=.02)
    ##################################Vision_Mamba_Part################################################
    vim_net = vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(class_num=3, drop_rate=0.1)
    weight_file = torch.load(opt.vim_path)
    weight_file = {k: v for k, v in weight_file.items() if (k in weight_file and 'head' not in k)}
    print(weight_file)
    vim_net.load_state_dict(weight_file['model'], strict=True)
    nn.init.trunc_normal_(vim_net.head.weight, std=.02)
    ##################################VMamba_Part################################################
    mamba_base = VSSM(patch_size=4, in_chans=3, num_classes=class_num, depths=[2, 2, 4, 2],
                      dims=[96, 192, 384, 768], ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                      ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v3noz",
                      mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False, drop_path_rate=0.2,
                      patch_norm=True, norm_layer="LN",  # "BN", "LN2D"
                      downsample_version="v3",  # "v1", "v2", "v3"
                      patchembed_version="v2",  # "v1", "v2"
                      use_checkpoint=False,
                      )
    weight_file = torch.load(opt.vmamba_path, map_location='cpu')["model"]
    weight_file = {k: v for k, v in weight_file.items() if (k in weight_file and 'classifier.head' not in k)}
    mamba_base.load_state_dict(weight_file, strict=False)"""