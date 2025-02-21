
import os.path
from configparser import ConfigParser
from argparse import ArgumentParser
import torch
import torch.optim as optim
from model import make_model
from utils import data_procession, get_config_of_layers, load_data_from_file,get_encoder_decoder_input_target
from train import trainer

config = ConfigParser()
config.read('ASTGNN.ini')
argp = ArgumentParser(description='This is for ASGTNN')
# 采用两种方式来进行参数的输入
argp.add_argument('data_path', help='this is used to load the original data')
argp.add_argument('mode', help='this is used to choose whether train or test', type=str)
argp.add_argument('--save_params_to', default=None, help='this is used to decide where to save the params')
argp.add_argument('--save_tensorboard_path', default=None,
                  help='this is used to decide where to save the data of tensorboard')
argp.add_argument('--batch_size', default=16, help='this is used to choose the batch size of data', type=int)
argp.add_argument('--epochs', default=100, type=int)
argp.add_argument('--lr', default=0.001, type=float, help='learning rate')
argp.add_argument('--is_shuffle', default=False, help='decide whether shuffle the dataset or not')
argp.add_argument('--use_distribute', default=False, help='decide whether to use the distributed training')
argp.add_argument('--save_prediction_to', default='prediction_result')
argp.add_argument('--loss_history_path', default='loss_history_path')
argp.add_argument('--dtype', default=torch.float32)
argp.add_argument('--adja_path', default='../data/PEMS08/distance.csv', type=str)
argp.add_argument('--pretrain_model_params_path', default=None, type=str)
argp.add_argument('--save', default=False, type=bool)
argp.add_argument('--merge', default=False, type=bool)

argp.add_argument('--SE', default=True, type=bool, help='this is used to choose whether to use spatial embedding')
argp.add_argument('--TE', default=True, type=bool)
argp.add_argument('--num_of_weeks', default=1, type=int)
argp.add_argument('--num_of_days', default=1, type=int)
argp.add_argument('--num_of_hours', default=1, type=int)
argp.add_argument('--points_per_hour', default=12, type=int)
argp.add_argument('--d', default=64, type=int)
argp.add_argument('num_for_predict', default=12, type=int)
argp.add_argument('--num_for_final_output', default=1, type=int)
argp.add_argument('--dropout', default=0.5, type=int)
argp.add_argument('--tr_self_att', default=True, type=bool)
argp.add_argument('--sdgcn_with_scale', default=True, type=bool)
argp.add_argument('--smooth_layers_num', default=1, type=int)

args = argp.parse_args()


def run(args):
    data_file_path = args.data_path
    num_of_weeks = args.num_of_weeks
    num_of_days = args.num_of_days
    num_of_hours = args.num_of_hours
    points_per_hour = args.points_per_hour
    shuffle = args.is_shuffle
    merge = args.merge
    ratio = (6, 8, 10)
    save = args.save
    num_of_decoder_layers = args.num_of_decoder_layers
    num_of_encoder_layers = args.num_of_encoder_layers
    d = args.d
    num_for_predict = args.num_for_predict
    num_for_final_output = args.num_for_final_output
    smooth_layers_num = args.smooth_layers_num
    sdgcn_with_scale = args.sdgcn_with_scale
    tr_self_att = args.tr_self_att
    TE = args.TE
    SE = args.SE
    dropout = args.dropout

    dataset, src_size = data_procession(data_file_path=data_file_path, num_of_weeks=num_of_weeks,
                                        num_of_days=num_of_days,
                                        num_of_hours=num_of_hours, points_per_hours=points_per_hour, shuffle=shuffle,
                                        merge=merge, ratio=(6, 8, 10), save=save)
    dataset=get_encoder_decoder_input_target(dataset)
    norm_adja = dataset['norm_adja']
    encoder_configs, decoder_configs = get_config_of_layers(args=args, norm_adja=norm_adja)

    model = make_model(decoder_configs=decoder_configs, encoder_configs=encoder_configs,
                       num_of_decoder_layers=num_of_decoder_layers,
                       src_size=src_size, num_of_encoder_layers=num_of_encoder_layers, N=src_size, d=d,
                       num_for_predict=num_for_predict, num_for_final_output=num_for_final_output,
                       smooth_layers_num=smooth_layers_num, norm_adja=norm_adja, sdgcn_with_scale=sdgcn_with_scale,
                       tr_self_att=tr_self_att, TE=TE, SE=SE, dropout=dropout
                       )
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.modules.MSELoss()
    batch_size=args.batch_size
    epochs=args.epochs
    save_params_to=args.save_params_to
    loss_history_path=args.loss_history_path
    use_distributed=args.use_distributed
    if save_params_to is None:
        file_name=os.path.basename(data_file_path).split('.')[0]
        dir_name=os.path.dirname(data_file_path)
        save_params_to=os.path.join(dir_name,f'model_params_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}{file_name}')
    if loss_history_path is None:
        file_name=os.path.basename(data_file_path).split('.')[0]
        dir_name=os.path.dirname(data_file_path)
        save_params_to=os.path.join(dir_name,f'loss_history_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}{file_name}')


    trainer(model=model,dataset=dataset,optimizer=optimizer,loss_fn=loss_fn,batch_size=batch_size,
            epochs=epochs,save_params_to=save_params_to,loss_history_path=loss_history_path,shuffle=shuffle,use_distributed=use_distributed)



if __name__=='__main__':
    run(args)