from chainer.links import ResNet50Layers


class MyResNet50Layers(ResNet50Layers):

    def __init__(self, *args, **kwargs):
        self.keys_to_remove = kwargs.pop('keys_to_remove', [])
        super().__init__(*args, **kwargs)

    @property
    def functions(self):
        funcs = super().functions
        for key in self.keys_to_remove:
            del funcs[key]
        return funcs


