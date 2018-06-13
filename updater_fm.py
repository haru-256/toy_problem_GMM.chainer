import chainer
from chainer import Variable
import chainer.functions as F


class DCGANUpdater(chainer.training.StandardUpdater):
    """
    costomized updater for DCGAN applied feature matching

    Upedater の自作は，基本的に__init__とupdate_core overrideすればよい
    """

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")  # extract model
        self.lam = kwargs.pop("lam")
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

        # transform to chainer.Variable and normalized to the range of -1 to 1
        # self.converterのdevice引数でgpuをしていすることで
        # データをgpuの形式に変更する->つまりcupyにする

        x_real = Variable(self.converter(
            batch, self.device))  # self.converter() is concat_example()
        x_real = (x_real - 127.5) / 127.5

        xp = chainer.backends.cuda.get_array_module(
            x_real.data)  # return cupy or numpy based on type of x_real.data

        gen, dis = self.gen, self.dis
        batch_size = len(batch)

        y_real, real_fm = dis(x_real)  # 本物の画像の推定結果
        real_fm_expected = F.mean(real_fm, axis=0)  # 本物のデータの場合のfeatureの期待値を近似

        z = Variable(xp.asarray(
            gen.make_hidden(batch_size)))  # genertate z random vector  xp.asarrayでcupy形式に変更する

        x_fake = gen(z)  # genertate fake data by generator

        y_fake, fake_fm = dis(x_fake)  # 偽物の画像の推定結果
        fake_fm_expected = F.mean(fake_fm, axis=0)  # 本物のデータの場合のfeatureの期待値を近似

        # dis, genをそれぞれ最適化
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        """
        gen_optimizer.update(self.loss_gen_fm, gen,
                             real_fm_expected, fake_fm_expected)  # gen_fmのLossを使用
        """
        gen_optimizer.update(self.loss_gen, gen,
                             y_fake, real_fm_expected, fake_fm_expected, self.lam)  # fm + original

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)  # dis/loss でアクセス可能

        return loss

    def loss_gen(self, gen, y_fake, real_fm_expected, fake_fm_expected, lam):
        fm_loss = F.mean_squared_error(
            real_fm_expected, fake_fm_expected)  # feature matching Loss
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize + lam*fm_loss
        chainer.report({'loss': loss}, gen)  # gen/loss でアクセス可能

        return loss

    def loss_gen_fm(self, gen, real_fm_expected, fake_fm_expected):
        fm_loss = F.mean_squared_error(real_fm_expected, fake_fm_expected)
        chainer.report({'fm_loss': fm_loss}, gen)  # gen/loss でアクセス可能

        return fm_loss
