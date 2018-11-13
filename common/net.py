import chainer
import chainer.functions as F
import chainer.links as L


class DownResBlock1(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock1, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(None, ch, 3, 1, 1, initialW=w, nobias=True)
            self.c1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w, nobias=True)
            self.cs = L.Convolution2D(None, ch, 4, 2, 1, initialW=w, nobias=True)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(self.h0)
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4


class DownResBlock2(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock2, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True)
            self.c1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w, nobias=True)
            self.cs = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w, nobias=True)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h3 = self.cs(self.h0)
        self.h4 = self.h2 + self.h3
        return self.h4


class DownResBlock3(chainer.Chain):
    """
        pre activation residual block
    """

    def __init__(self, ch):
        w = chainer.initializers.Normal(0.02)
        super(DownResBlock3, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, nobias=True)

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.relu(self.h0))
        self.h2 = self.c1(F.relu(self.h1))
        self.h4 = self.h2 + self.h0
        return self.h4


class ResnetAssessor(chainer.Chain):
    def __init__(self, bottom_width=8, ch=128, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(ResnetAssessor, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        with self.init_scope():
            self.r0 = DownResBlock1(128)
            self.r1 = DownResBlock2(128)
            self.r2 = DownResBlock3(128)
            self.r3 = DownResBlock3(128)
            self.l4 = L.Linear(None, output_dim, initialW=w, nobias=True)

    def __call__(self, x):
        self.x = x
        self.h1 = self.r0(self.x)
        self.h2 = self.r1(self.h1)
        self.h3 = self.r2(self.h2)
        self.h4 = self.r3(self.h3)
        h = F.relu(self.h4)
        return F.sigmoid(self.l4(h))
