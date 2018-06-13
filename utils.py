import pathlib
import numpy as np
import seaborn as sns
import chainer
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import chainer.backends.cuda
from chainer import Variable
sns.set()


def out_generated(gen, seed, dst, datasize=1000, **kwards):
    """
    Trainer extension that plot Generated data

    Parameters
    -------------
    gen: Model
        Generator

    seed: int
        fix random by value

    dst: PosixPath
        file path to save plotted result

    datasize: int
        the number of plotted datas

    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot

    Return
    ------------
    make_image:function
        function that returns make_images that has Trainer object
        as argument.
    """
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):

        np.random.seed(seed)  # fix seed
        xp = gen.xp  # get module

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                z = Variable(xp.asarray(gen.make_hidden(datasize)))
                x = gen(z)

        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        preview_dir = pathlib.Path('{}/preview'.format(dst))
        preview_path = preview_dir /\
            '{:}epoch_kde.jpg'.format(trainer.updater.epoch)
        if not preview_dir.exists():
            preview_dir.mkdir()
        # norm = Normalize(vmin=x.data.min(),  vmax=x.data.max())  # colorbar range of kde
        plot_kde_data(x, trainer.updater.epoch, preview_path, shade=True, cbar=False,
                      cmap="Blues", shade_lowest=False, **kwards)
        preview_path = preview_dir /\
            '{:}epoch_scatter.jpg'.format(trainer.updater.epoch)
        plot_scatter_data(x, trainer.updater.epoch, preview_path, **kwards)

    return make_image


def plot_scatter_data(data, epoch, preview_path, **kwards):
    """
    Plot the data

    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data

    epoch: int
        Epoch number

    preview_path: PosixPath
        file path to save plotted results

    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.scatter(data[:, 0], data[:, 1],
                 alpha=0.5, color='darkgreen', s=17)
    axes.set_title('epoch: {:>3}'.format(epoch))
    axes.set_xlim(-radius-1.5, radius+1.5)
    axes.set_ylim(-radius-1.5, radius+1.5)
    fig.tight_layout()
    fig.savefig(preview_path)
    plt.close(fig)


def plot_kde_data(data, epoch, preview_path, **kwards):
    """
    Plot the data

    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data

    epoch: int
        Epoch number

    preview_path: PosixPath
        file path to save plotted result

    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes = sns.kdeplot(data=data[:, 0], data2=data[:, 1],
                       ax=axes, **kwards)
    axes.set_title('epoch: {:>3}'.format(epoch))
    axes.set_xlim(-radius-1.5, radius+1.5)
    axes.set_ylim(-radius-1.5, radius+1.5)
    fig.tight_layout()
    fig.savefig(preview_path)
    plt.close(fig)


def plot_kde_data_real(data, file_path, **kwards):
    """
    Plot the data

    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data

    epoch: int
        Epoch number

    file_path: PosixPath
        file path to save plotted result

    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    # norm = Normalize(vmin=0,  vmax=1)  # colorbar range of kde
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes = sns.kdeplot(data=data[:, 0], data2=data[:, 1],
                       ax=axes, **kwards)
    axes.set_title('Real Data')
    axes.set_xlim(-radius-1.5, radius+1.5)
    axes.set_ylim(-radius-1.5, radius+1.5)
    fig.tight_layout()
    fig.savefig(file_path / 'training_data_kde.jpg')
    plt.close(fig)


def plot_scatter_real_data(data, file_path, **kwards):
    """
    Plot the data

    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data

    preview_path: PosixPath
        file path to save plotted results

    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.scatter(data[:, 0], data[:, 1],
                 alpha=0.5, color='darkgreen', s=17)
    axes.set_title('Real Data')
    axes.set_xlim(-radius-1.5, radius+1.5)
    axes.set_ylim(-radius-1.5, radius+1.5)
    fig.tight_layout()
    fig.savefig(file_path / 'training_data_scatter.jpg')
    plt.close(fig)
