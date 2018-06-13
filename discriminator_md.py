import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class Minibatch_Discrimination(chainer.Chain):
    """
    Minibatch Discrimination Layer

    Parameters
    ---------------------
    B: int
        number of rows of M

    C: int
        number of columns of M

    wscale: float
        std of normal initializer

    """

    def __init__(self, B, C, wscale):
        super(Minibatch_Discrimination, self).__init__()
        self.b = B
        self.c = C
        with self.init_scope():
            # initialozer to W
            w = chainer.initializers.Normal(wscale)

            # register Parameters
            self.t = L.Linear(in_size=None,
                              out_size=B*C,
                              initialW=w,
                              nobias=True)  # bias is required ?

    def __call__(self, x):
        """
        Calucurate Minibatch Discrimination using broardcast.

        Parameters
        ---------------
        x: Variable
           input vector shape is (N, num_units)
        """
        batch_size = x.shape[0]
        xp = x.xp
        activation = self.t(x)
        m = F.reshape(activation, (-1, self.b, self.c))
        m = F.expand_dims(m, 3)
        m_T = F.transpose(m, (3, 1, 2, 0))
        m, m_T = F.broadcast(m, m_T)
        l1_norm = F.sum(F.absolute(m-m_T), axis=2)

        # eraser to erase l1 norm with themselves
        eraser = F.expand_dims(xp.eye(batch_size, dtype="f"), 1)
        eraser = F.broadcast_to(eraser, (batch_size, self.b, batch_size))

        o_X = F.sum(F.exp(-(l1_norm + 1e6 * eraser)), axis=2)

        # concatunate along channels or units
        return F.concat((x, o_X), axis=1)


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

    def __init__(self, in_ch=1, wscale=0.02, B=32, C=8):
        super(Discriminator, self).__init__()
        self.b, self.c = B, C
        with self.init_scope():
            # initializers
            w = chainer.initializers.HeNormal(wscale)

            # register layer with variable
            self.l0 = L.Linear(in_size=None, out_size=128, initialW=w)
            self.md1 = Minibatch_Discrimination(
                B=self.b, C=self.c, wscale=wscale)
            self.l2 = L.Linear(in_size=None, out_size=1, initialW=w)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = self.md1(h)
        logits = self.l2(h)

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
