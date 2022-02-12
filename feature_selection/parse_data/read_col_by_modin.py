from ray.data import Dataset
from ray.data.impl.arrow_block import ArrowRow


def dataset2modin(ds: Dataset[ArrowRow]):
    modin_ds = ds.to_modin()
    return modin_ds
