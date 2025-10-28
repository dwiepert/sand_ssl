"""
TODO
"""
from typing import Union
from pathlib import Path
import torch

def save_checkpoint(model, path:Union[str,Path], sub_dir:str):
        """
        Save checkpoint

        :param path: pathlike, Path to save checkpoint at (FULL PATH)
        """
        if not isinstance(path, Path): path = Path(path)

        if sub_dir: 
            path = path / sub_dir
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        name_model = model.get_model_name()
        path = path / f'{model.get_model_name()}.pth'

        assert path.suffix == '.pth' or path.suffix == '.pt', 'Must give a full .pt or .pth filepath'

        torch.save(model.state_dict(), path)
        return