from .libero import parse_meta_libero

DATASET_PARSERS_META = {
    "libero": parse_meta_libero,#官方的libero数据集格式：.h5文件或者.hdf5文件
}
