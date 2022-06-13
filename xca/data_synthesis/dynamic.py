import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from xca.data_synthesis.builder import single_pattern
from xca.data_synthesis.cctbx import load_cif, calc_structure_factor, convert_to_numpy
from pathlib import Path
from typing import Optional, Callable


def get_reflections(cif_path, tth_min, tth_max, wavelength):
    """Checks relevant reflections that occur between tth_min and tth_max at a given wavelength"""
    data = load_cif(cif_path)
    sf = calc_structure_factor(data["structure"])
    scattering = convert_to_numpy(
        sf, wavelength=wavelength, tth_max=tth_max, tth_min=tth_min
    )
    reflections = zip(scattering.hkl, scattering.tth, scattering.I)
    keep = []
    for reflection in reflections:
        if tth_max > reflection[1] > tth_min and reflection[2] > 1:
            keep.append(tuple(reflection[0]))
    return keep


def log_reflections(cif_paths, tth_min, tth_max, wavelength):
    """Itterates over list of cifs and puts relevant reflections into json file"""
    dic = {}
    for cif_path in cif_paths:
        if not isinstance(cif_path, Path):
            path = Path(cif_path)
        else:
            path = cif_path
        dic[path.stem] = get_reflections(path, tth_min, tth_max, wavelength)
    return dic


class DynamicTrainingDataset(Dataset):
    def __init__(
        self,
        *,
        cif_paths: list[Path],
        param_dict: dict,
        shape_limit: float,
        target: Optional[str] = None,
        target_transform: Optional[Callable] = None,
        epoch_len: Optional[int] = None,
        **kwargs
    ):
        """

        Parameters
        ----------
        cif_paths : list[Path]
        param_dict : dict
            Dictionary of parameters for cctbx wrappers
        shape_limit : float
            Limit on shape modulation
        target : Optional[str]
            String target for prediction. Defaults to a classification of the phases by path stem
        target_transform : Optional[Callable]
            Callable that returns the appropriate tensor from the da.attrs[target]. This could for e.g.
            include normalization or dtype enforcement.
        epoch_len : Optional[int]
            Number of samples per epoch. If none given, it defaults to 10 times the number of cif paths.
        kwargs
        """
        self.param_dict = param_dict
        self.wavelength = param_dict["wavelength"]
        self.tth_range = (param_dict["2theta_min"], param_dict["2theta_max"])
        self.phases = [path.stem for path in cif_paths]
        self.cifs = {path.stem: path for path in cif_paths}
        self.reflections = log_reflections(
            cif_paths, self.tth_range[0], self.tth_range[1], self.wavelength
        )
        self.n_phases = len(cif_paths)
        self.shape_limit = shape_limit
        self.synth_kwargs = kwargs

        if target is None:
            self.target = "input_cif"
            if target_transform is None:
                self.phase_dict = {phase: i for i, phase in enumerate(self.phases)}
                self.target_transform = self._default_class_transform
        elif target_transform is None:
            self.target_transform = self._default_transform

        if epoch_len is None:
            self.epoch_len = 10 * len(self.phases)
        else:
            self.epoch_len = epoch_len

    def _default_class_transform(self, y):
        return torch.tensor(self.phase_dict[y], dtype=torch.long)

    @staticmethod
    def _default_transform(y):
        return torch.tensor(y)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        idx = idx % self.n_phases
        phase = self.phases[idx]
        cif_path = self.cifs[phase]
        _param_dict = {"input_cif": cif_path}
        _param_dict.update(self.param_dict)
        da = single_pattern(
            _param_dict, shape_limit=self.shape_limit, **self.synth_kwargs
        )
        return (
            torch.tensor(da.data[None, ...], dtype=torch.float),
            self.target_transform(da.attrs[self.target]),
        )


class DynamicDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 32,
        batch_per_train_epoch: int = 100,
        batch_per_val_epoch: int = 10,
        **kwargs
    ):
        """
        Lightning data module to manage dynamic dataset generation

        Parameters
        ----------
        batch_size : int
        num_workers : int
        batch_per_train_epoch : int
            Since epoch length is arbitrary since the dataset is constantly growing, a factor of batch
            size is used to determine the artificial length of an epoch.
        batch_per_val_epoch : int
            Since epoch length is arbitrary since the dataset is constantly growing, a factor of batch
            size is used to determine the artificial length of an epoch.
        kwargs
            keyword arguments passed to DynamicTrainDataset initialization
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = DynamicTrainingDataset(epoch_len=batch_per_train_epoch, **kwargs)
        self.val = DynamicTrainingDataset(epoch_len=batch_per_val_epoch, **kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )
