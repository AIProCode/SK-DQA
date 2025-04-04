from openvqa.models.model_loader import CfgLoader
from utils.exec import Execution
import argparse, yaml


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='OpenVQA Args')

    parser.add_argument('--RUN', dest='RUN_MODE',
                        default='train',
                        choices=['train', 'val', 'test'],
                        help='{train, val, test}',
                        type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                        default='skdqa',
                        type=str, required=True)

    parser.add_argument('--DATASET', dest='DATASET',
                        default='csdqa',
                        type=str, required=True)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                        default='train',
                        choices=['train', 'train+val', 'train+val+vg'],
                        help="set training split, vqa: {'train', 'train+val', 'train+val+vg'} clevr: {'train', 'train+val'}",
                        type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                        default='True',
                        choices=['True', 'False'],
                        help='True: evaluate the val split when an epoch finished, False: do not evaluate on local',
                        type=str)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                        default='False',
                        choices=['True', 'False'],
                        help='True: save the prediction vectors,'
                           'False: do not save the prediction vectors',
                        type=str)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                        default=32,
                        help='batch size in training',
                        type=int)

    parser.add_argument('--GPU', dest='GPU',
                        default='0',
                        help="gpu choose, eg.'0, 1, 2, ...'",
                        type=str)

    parser.add_argument('--SEED', dest='SEED',
                        default=627894,
                        help='fix random seed',
                        type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                        default='vqabase',
                        help='version control',
                        type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                        default='False',
                        choices=['True', 'False'],
                        help='True: use checkpoint to resume training,'
                           'False: start training with random init',
                        type=str)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                        default="vqabase",
                        help='checkpoint version',
                        type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                        default=20,
                        help='checkpoint epoch',
                        type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                        default='/root/ckpts',
                        help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead, it will override'
                           'CKPT_VERSION and CKPT_EPOCH',
                        type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                        help='split batch to reduce gpu memory usage',
                        type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                        default=4,
                        help='multithreaded loading to accelerate IO',
                        type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                        default='True',
                        choices=['True', 'False'],
                        help='True: use pin memory, False: not use pin memory',
                        type=str)

    parser.add_argument('--VERB', dest='VERBOSE',
                        default='True',
                        choices=['True', 'False'],
                        help='True: verbose print, False: simple print',
                        type=str)
    
    parser.add_argument('--Ctype', dest='USE_CONTEXT',
                        default='None',
                        choices=['None', 'deep', 'global', 'deep-global'],
                        help='choose context style for CAAN',
                        type=str)
    
    parser.add_argument('--LG', dest='Lang_Global',
                        default='max',
                        choices=['max', 'avg', 'att', 'multimodal'],
                        help='choose global language feature style for MTCCT',
                        type=str)
    
    parser.add_argument('--REL', dest='REL',
                        choices=['True', 'False'],
                        help='choose whether REL style for CAANplus',
                        type=str)
    
    parser.add_argument('--Comp', dest='COMPACT',
                        choices=['False', 'True'],
                        help='choose whether use compact SDPA for CLVIN',
                        type=str)
                      
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cfg_file = "/root/configs/{}/{}.yml".format(args.DATASET, args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.CLoader)

    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    args = __C.str_to_bool(args)
    args_dict = __C.parse_to_dict(args)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)




