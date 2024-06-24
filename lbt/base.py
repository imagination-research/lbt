# -*- coding: utf-8 -*-
# Copied from https://github.com/walkerning/aw_nas

from collections import OrderedDict
import six
from six import StringIO
import yaml

from lbt import utils
from lbt.utils import RegistryMeta
from lbt.utils import getLogger

# Make yaml.safe_dump support OrderedDict
yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
    Dumper=yaml.dumper.SafeDumper,
)


LOGGER = getLogger("registry")


@six.add_metaclass(RegistryMeta)
class Component:
    def __init__(self):
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # set self._logger to None
        self._logger = None

    @classmethod
    def get_default_config(cls):
        return utils.get_default_argspec(cls.__init__)

    @classmethod
    def get_default_config_str(cls):
        stream = StringIO()
        cfg = OrderedDict(cls.get_default_config())
        yaml.safe_dump(cfg, stream=stream, default_flow_style=False)
        return stream.getvalue()

    @classmethod
    def get_current_config_str(cls, cfg):
        stream = StringIO()
        whole_cfg = OrderedDict(cls.get_default_config())
        whole_cfg.update(cfg)
        yaml.safe_dump(whole_cfg, stream=stream, default_flow_style=False)
        return stream.getvalue()

    @classmethod
    def init_from_cfg_file(cls, cfg_path, registry_name=None, **addi_kwargs):
        with open(cfg_path, "r") as rf:
            cfg = yaml.safe_load(rf)
        return cls.init_from_cfg(cfg, registry_name=registry_name, **addi_kwargs)

    @classmethod
    def init_from_cfg(cls, cfg, registry_name=None, **addi_kwargs):
        avail_registries = RegistryMeta.avail_tables()
        if not hasattr(cls, "REGISTRY"):
            # Component class
            if registry_name is not None:
                assert registry_name in avail_registries
            else:
                type_keys = [
                    key
                    for key in cfg.keys()
                    if key.endswith("_type") and key[:-5] in avail_registries
                ]
                assert len(type_keys) == 1
                registry_name = type_keys[0][:-5]
                LOGGER.info(f"Guess `registry_name={registry_name}` from the config.")
        elif not hasattr(cls, "NAME"):
            # Base classes that inherit `Component`
            assert registry_name is None or registry_name == cls.REGISTRY, (
                f"This class `{cls.__name__}` is in registry `{cls.REGISTRY}`, do not"
                f" match `{registry_name}`. Either do not pass in the `registry_name`"
                " argument or pass in a matching registry name."
            )
            registry_name = cls.REGISTRY
            type_ = cfg[registry_name + "_type"]
            true_cls = RegistryMeta.get_class(registry_name, type_)
        else:
            # Concrete class
            assert registry_name is None or registry_name == cls.REGISTRY, (
                f"This class `{cls.__name__}` is in registry `{cls.REGISTRY}`, do not"
                f" match `{registry_name}`. Either do not pass in the `registry_name`"
                " argument or pass in a matching registry name."
            )
            registry_name = cls.REGISTRY
            type_ = cls.NAME

        type_ = cfg[registry_name + "_type"]
        true_cls = RegistryMeta.get_class(registry_name, type_)
        LOGGER.info(
            "Component [%s] type： %s, %s.%s",
            registry_name,
            type_,
            true_cls.__module__,
            true_cls.__name__,
        )

        class_cfg = cfg.get(registry_name + "_cfg", {})
        class_cfg = class_cfg or {}
        # config items will override addi_args items
        addi_kwargs.update(class_cfg)

        whole_cfg_str = true_cls.get_current_config_str(class_cfg)
        LOGGER.info(
            "%s `%s` config:\n%s",
            registry_name,
            type_,
            utils._add_text_prefix(whole_cfg_str, "  "),
        )
        return true_cls(**addi_kwargs)
