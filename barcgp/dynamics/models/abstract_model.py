#!/usr/bin python3

from abc import ABC, abstractmethod
import pathlib
import shutil
import os

import casadi as ca

from barcgp.dynamics.models.model_types import ModelConfig

class AbstractModel(ABC):
    '''
    Base class for models
    Controllers may differ widely in terms of algorithm runtime and setup however
      for interchangeability controllers should implement a set of standard runtime methods:
    '''
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        
        if not model_config.enable_jacobians:
            jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        else:
            jac_opts = dict()
        self.options = lambda fn_name: dict(jit=False, **jac_opts)

    @abstractmethod
    def step(self):
        pass

    # Method for installing generated files
    def install(self, dest_dir: str=None, src_dir: str=None, verbose=False):
        # If no target directory is provided, try to install a directory with
        # the same name as the model name in the current directory
        if src_dir is None:
            src_path = pathlib.Path.cwd().joinpath(self.model_config.model_name)
        else:
            src_path = pathlib.Path(src_dir).expanduser()

        if dest_dir is None:
            if self.model_config.install_dir is None:
                if verbose:
                    print('- No destination directory provided, did not install')
                return None
            dest_path = pathlib.Path(self.model_config.install_dir).expanduser()
        else:
            dest_path = pathlib.Path(dest_dir).expanduser()

        if src_path.exists():
            if not dest_path.exists():
                dest_path.mkdir(parents=True)
            # If directory with same name as model already exists, delete
            if dest_path.joinpath(self.model_config.model_name).exists():
                if verbose:
                    print('- Existing installation found, removing...')
                shutil.rmtree(dest_path.joinpath(self.model_config.model_name))
            shutil.move(str(src_path), str(dest_path))
            if verbose:
                print('- Installed files from source: %s to destination: %s' % (str(src_path), str(dest_path.joinpath(self.model_config.model_name))))
            return dest_path.joinpath(self.model_config.model_name)
        else:
            if verbose:
                print('- The source directory %s does not exist, did not install' % str(src_path))
            return None
