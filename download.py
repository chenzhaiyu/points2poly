"""
Download datasets and/or models from public urls.
"""

import os
import zipfile
import urllib.request

from tqdm import tqdm
import hydra
from omegaconf import DictConfig


def my_hook(t):
    """
    Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py

    Example
    -------
        with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def download(cfg: DictConfig):
    """
    Download datasets and/or models from public urls.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """
    data_url, model_url = cfg.url_datasets[cfg.dataset_name], cfg.url_models[cfg.model_name]
    data_dir, model_dir = cfg.datadir, cfg.modeldir
    data_file, model_file = os.path.join(data_dir, f'{cfg.dataset_name}.zip'), os.path.join(model_dir,
                                                                                            f'{cfg.model_name}.zip')

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=data_file) as t:
        urllib.request.urlretrieve(data_url, filename=data_file, reporthook=my_hook(t), data=None)

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=model_file) as t:
        urllib.request.urlretrieve(model_url, filename=model_file, reporthook=my_hook(t), data=None)

    # unzipping
    zip_data, zip_model = zipfile.ZipFile(data_file, 'r'), zipfile.ZipFile(model_file, 'r')
    zip_data.extractall(data_dir)
    zip_model.extractall(model_dir)
    zip_data.close()
    zip_model.close()
    os.remove(data_file)
    os.remove(model_file)


if __name__ == '__main__':
    download()
