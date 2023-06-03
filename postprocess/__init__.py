import copy

__all__ = ['build_post_process']

from .db_postprocess import DBPostProcess
from .rec_postprocess import CTCLabelDecode, AttnLabelDecode
from .cls_postprocess import ClsPostProcess
from .table_postprocess import  TableLabelDecode, TableMasterLabelDecode


def build_post_process(config, global_config=None):
    support_dict = ['DBPostProcess', 'CTCLabelDecode', 'AttnLabelDecode', 'ClsPostProcess','DistillationCTCLabelDecode', 'TableLabelDecode', 'TableMasterLabelDecode']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
