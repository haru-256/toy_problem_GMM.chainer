import chainer
from chainer import Variable
import chainer.functions as F


class DCGANUpdater(chainer.training.StandardUpdater):
    """
    costomized updater for DCGAN

    Upedater の自作は，基本的に__init__とupdate_core overrideすればよい
    """

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")  # extract model
        self.scale = kwargs.pop('scale')
        super(DCGANUpdater, self).__init__(*args,
                                           **kwargs)  # StandardUpdaterを呼ぶ

    def update_core(self):
        # get_optimizer mehtod allows to get optimizer
        # The name "main" is valid when there is one optimizer
        gen_optimizer = self.get_optimizer("gen")
        dis_optimizer = self.get_optimizer("dis")
        # obtain batch data
        # get_iterator("main") is SerialIterator so next() returns next minibatch
        batch = self.get_iterator("main").next()

        x_real = Variable(self.converter(
            batch, self.device))  # self.converter() is concat_example()
        # x_real = x_real / self.scale

        xp = chainer.backends.cuda.get_array_module(
            x_real.data)  # return cupy or numpy based on type of x_real.data

        gen, dis = self.gen, self.dis
        batch_size = len(batch)

        y_real = dis(x_real)  # 本物の画像の推定結果

        z = Variable(xp.asarray(
            gen.make_hidden(batch_size)))  # genertate z random vector  xp.asarrayでcupy形式に変更する

        x_fake = gen(z)  # genertate fake data by generator

        y_fake = dis(x_fake)  # 偽物の画像の推定結果

        # dis, genをそれぞれ最適化
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)  # dis/loss でアクセス可能
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)  # gen/loss でアクセス可能
        return loss
