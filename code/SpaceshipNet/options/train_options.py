from .base_options import BaseOptions
from skyu_tools import  skyu_util as sutil
import os

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')       
        self.parser.add_argument('--verbose_fre', type=int, default=200, help='verbose_fre')       
        self.parser.add_argument('--synthesize_interval', type=int, default=1, help='synthesize_interval 2 epochs')       
        self.parser.add_argument('--G_interge_epoch', type=bool, default=True, help='synthesize_interval 2 epochs')       
        self.parser.add_argument('--DDP_batchsize_split', type=bool, default=True, help='split batchsize when using DDP')       
        self.parser.add_argument('--aug_padding', type=int, default=4, help='padding used by self.aug')       
        self.parser.add_argument('--generate_input_test', type=bool, default=True, help='generate_input_test')       
        self.parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
        self.isTrain = True

