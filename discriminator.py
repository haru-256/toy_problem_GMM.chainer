import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    """Discriminator

    build Discriminator model

    Parametors
    ---------------------
    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    wscale: float
        std of normal initializer
    Attributes
    ---------------------
    """

    def __init__(self, in_ch=1, wscale=0.02):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # initializers
            w = chainer.initializers.HeNormal(wscale)

            # register layer with variable
            self.l0 = L.Linear(in_size=None, out_size=128, initialW=w)
            self.l1 = L.Linear(in_size=None, out_size=1, initialW=w)

            self.bn1 = L.BatchNormalization(size=128)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        logits = self.l1(h)

        return logits


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    z = np.random.uniform(-1, 1, (10, 2)).astype("f")
    model = Discriminator()
    img = model(Variable(z))
    # print(img)
    g = c.build_computational_graph(img)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
