import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    """Discriminator

    build Discriminator model applied feature matching

    Parametors
    ---------------------
    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    wscale: float
        std of normal initializer
    Attributes
    ---------------------

    Returns
    --------------------
    y: float
        logits
    h: float
        feature of one befor the out layer
    """

    def __init__(self, in_ch=1, wscale=0.02):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # initializers
            w = chainer.initializers.Normal(wscale)

            # register layer with variable
            self.c0 = L.Convolution2D(
                in_channels=in_ch,
                out_channels=64,
                ksize=4,
                stride=2,
                pad=1,
                initialW=w)
            self.c1 = L.Convolution2D(
                in_channels=None,
                out_channels=128,
                ksize=4,
                stride=2,
                pad=1,
                initialW=w,
                nobias=True)
            self.l2 = L.Linear(in_size=None, out_size=1, initialW=w)

            # self.bn1 = L.BatchNormalization(size=128)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        y = self.l2(h)  # conv->linear では勝手にreshapeが適用される

        return y, h  # Also return feature


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    z = np.random.uniform(-1, 1, (1, 1, 28, 28)).astype("f")
    model = Discriminator()
    img = model(Variable(z))
    # print(img)
    g = c.build_computational_graph(img)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
