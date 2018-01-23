from matplotlib import pyplot
import os.path


def plot_tsne(X, y, f_name):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=pyplot.cm.viridis)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    # pyplot.show(block=False)
    pyplot.savefig(f_name + '.png')


def plot_losses(loss, val_loss):
    fig = pyplot.figure()
    pyplot.plot(loss)
    pyplot.plot(val_loss)
    pyplot.show(block=True)
    pyplot.savefig('loss.png')


def args_to_string(args):
    all_args = [str(k) + str(v) for k, v in vars(args).items()]
    str_args = '_'.join(str(arg) for arg in all_args)
    return str_args


def make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
