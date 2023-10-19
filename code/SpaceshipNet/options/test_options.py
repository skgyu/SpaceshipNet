from .base_options import BaseOptions
from skyu_tools import  skyu_util as sutil
import os

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.isTrain = False

        


