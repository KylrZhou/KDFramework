# Modified from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/__init__.py#L18

class Registry():
    def __init__(self, obj_type):
        self._obj_type = obj_type
        self._obj_map = {}

    def _do_register(self, TYPE, obj):
        assert (TYPE not in self._obj_map), (f"An object TYPE '{TYPE}' was already registered "
                                             f"in '{self._TYPE}' registry!")
        self._obj_map[TYPE] = obj

    def register(self, obj=None):
        if obj is None:
            def deco(func_or_class):
                TYPE = func_or_class.__name__
                self._do_register(TYPE, func_or_class)
                return func_or_class
            return deco
        TYPE = obj.__name__
        self._do_register(TYPE, obj)

    def get(self, TYPE):
        ret = self._obj_map.get(TYPE)
        if ret is None:
            raise KeyError(f"No object TYPE '{TYPE}' found in '{self._TYPE}' registry!")
        return ret

    def __contains__(self, TYPE):
        return TYPE in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()

MODEL = Registry('model')
PIPELINE = Registry('pipeline')
DATASETTRAIN = Registry('dataset_train')
DATASETVAL = Registry('dataset_val')
AUGMENTATION = Registry('augmentation')
OPTIMIZER = Registry('optimizer')
SCHEDULER = Registry('scheduler') 
LOGGER = Registry('logger')
FUNCTION = Registry('function')
DISTILLER = Registry('distiller')