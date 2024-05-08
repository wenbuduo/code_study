# 写日志
import logging


logging.basicConfig(level = logging.NOTSET, format = '%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger('Log')
