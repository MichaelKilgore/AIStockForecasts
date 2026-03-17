
import logging

logger = logging.getLogger(__name__)
INFO_VERBOSE = 19

def verbose_log(msg: object):
    logger.log(INFO_VERBOSE, msg)

