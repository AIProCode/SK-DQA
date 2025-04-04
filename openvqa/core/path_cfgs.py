import os

class PATH:
    def __init__(self):
        self.init_path()
        # self.check_path()


    def init_path(self):

        self.DATA_ROOT = '/root'


        self.DATA_PATH = {
            'vqa': self.DATA_ROOT + '/TQA',
            'vqa_1': self.DATA_ROOT + '/VQA2.0'
        }


        self.FEATS_PATH = {

            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/train',
                'val': self.DATA_PATH['vqa'] + '/val',
                'test': self.DATA_PATH['vqa'] + '/test',
            },
            'vqa_patch': {
                'train': self.DATA_PATH['vqa'] + '/train-p',
                'val': self.DATA_PATH['vqa'] + '/val-p',
                'test': self.DATA_PATH['vqa'] + '/test-p',
            },
            'vqa_cap': {
                'train': self.DATA_PATH['vqa'] + '/train-c',
                'val': self.DATA_PATH['vqa'] + '/val-c',
                'test': self.DATA_PATH['vqa'] + '/test-c',
            },
            

        }

        self.RAW_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/train_question.json',
                'train-anno': self.DATA_PATH['vqa'] + '/train_annotation.json',
                'val': self.DATA_PATH['vqa'] + '/val_question.json',
                'val-anno': self.DATA_PATH['vqa'] + '/val_annotation.json',
                'vg': 'vg_questions.json',
                'vg-anno': 'vg_annotations.json',
                'test': self.DATA_PATH['vqa'] + '/test_question.json',
            },
        }


        self.SPLITS = {
            'vqa': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            'clevr': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            'vqa_grid': {
                'train': '',
                'val': 'val',
                'test': 'test',
            }

        }

        self.RESULT_PATH = './results/result_test'
        self.PRED_PATH = './results/pred'
        self.CACHE_PATH = './results/cache'
        self.LOG_PATH = './results/log'
        self.CKPTS_PATH = './ckpts'


    def check_path(self, dataset=None):
        print('Checking dataset ........')

        if dataset:
            for item in self.FEATS_PATH[dataset]:
                if not os.path.exists(self.FEATS_PATH[dataset][item]):
                    print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

            for item in self.RAW_PATH[dataset]:
                if not os.path.exists(self.RAW_PATH[dataset][item]):
                    print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

        else:
            for dataset in self.FEATS_PATH:
                for item in self.FEATS_PATH[dataset]:
                    if not os.path.exists(self.FEATS_PATH[dataset][item]):
                        print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)
                        
            for dataset in self.SPATIALS_PATH:
                for item in self.SPATIALS_PATH[dataset]:
                    if not os.path.exists(self.SPATIALS_PATH[dataset][item]):
                        print(self.SPATIALS_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

            for dataset in self.RAW_PATH:
                for item in self.RAW_PATH[dataset]:
                    if not os.path.exists(self.RAW_PATH[dataset][item]):
                        print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

        print('Finished!')
        print('')

