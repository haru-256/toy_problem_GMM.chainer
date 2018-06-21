import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    """Generator

    build Generator model

    Parametors
    ---------------------
    n_hidden: int
       dims of random vector z

    bottom_width: int
       Width when converting the output of the first layer
       to the 4-dimensional tensor

    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    Attributes
    ---------------------
    """

    def __init__(self, n_hidden=100, wscale=0.02, isBN=False):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.isBN = isBN
        with self.init_scope():
            # w = chainer.initializers.HeNormal(wscale)  # initializers
            w = chainer.initializers.Normal(wscale)
            b = chainer.initializers.Normal(wscale)

            if self.isBN:
                self.l0 = L.Linear(in_size=self.n_hidden,
                                   out_size=128, initialW=w, nobias=True)
                self.l1 = L.Linear(in_size=None, out_size=128,
                                   initialW=w, nobias=True)

                self.bn0 = L.BatchNormalization(size=128)
                self.bn1 = L.BatchNormalization(size=128)
            else:
                self.l0 = L.Linear(in_size=self.n_hidden,
                                   out_size=128, initialW=w, initial_bias=b)
                self.l1 = L.Linear(in_size=None, out_size=128,
                                   initialW=w, initial_bias=b)

            self.l2 = L.Linear(in_size=None, out_size=2,
                               initialW=w, initial_bias=b)

    def make_hidden(self, batchsize=100):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        batchsize: int
           batchsize indicate len(z)
        """
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden))\
                        .astype(np.float32)

    def __call__(self, z):
        if self.isBN:
            h = F.relu(self.bn0(self.l0(z)))
        else:
            h = F.relu(self.l0(z))
        if self.isBN:
            h = F.relu(self.bn1(self.l1(h)))
        else:
            h = F.relu(self.l1(h))
        x = self.l2(h)  # linear projection to 2

        return x


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable

    model = Generator(n_hidden=100, isBN=False)
    img = model(Variable(model.make_hidden(10)))
    # print(img)
    g = c.build_computational_graph(img)
    with open('gen_graph.dot', 'w') as o:
        o.write(g.dump())
