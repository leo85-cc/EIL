'''19/06/03 lpl 日志记录类'''
'''主要用于记录程序的输出'''
import logging
import os.path
import time
class logger:
    def __init__(self,path,logger_name):
        self.path=path
        self.name=logger_name
        self.log=logging.getLogger(logger_name)
        self.log.setLevel(logging.INFO)
        # creat hander
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_file_name = path + rq + '_' + logger_name + '.log'
        fh = logging.FileHandler(log_file_name, mode='w')
        self.log.addHandler(fh)
    def output(self,info):
        self.log.info(info)
if __name__=='__main__':
    print('create a logger...')
    log=logger('/home/lpl/lpl/logs/','acc')
    log.output((1,2.0,2.0256))

