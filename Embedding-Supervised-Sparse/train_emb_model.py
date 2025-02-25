import sys, os

from torch.utils.tensorboard import SummaryWriter
from absl import flags
from absl import app

from scripts.utils import save_to_logs, get_train_dir
from scripts.emb_model_lib import EmbeddingModel

wkdir = os.getcwd()
sys.path.append(wkdir)

FLAGS = flags.FLAGS


def main(argv):
    args = {
        'dataset': FLAGS.dataset,
        'model': FLAGS.model,
        'num_classes': FLAGS.num_classes,
        'batch': FLAGS.batch,
        'lr': FLAGS.lr,
        'embs': None # 'z'或'g'，用于区分两个阶段
    }

    args['embs'] = 'z'

    train_dir = get_train_dir(wkdir, args, 'emb_net')

    writer = SummaryWriter(train_dir + 'logs/')
    z_emb_model = EmbeddingModel(args, wkdir, writer)

    start_epoch = z_emb_model.load_from_checkpoint(mode='latest')
    valid_acc = z_emb_model.get_test_accuracy(return_acc=True)
    # x -> z -> e
    for epoch in range(start_epoch, 5):
        loss = z_emb_model.train_one_epoch(epoch)
        valid_acc = z_emb_model.get_test_accuracy(return_acc=True)
        print(f'loss: {loss}')
        save_to_logs(train_dir, valid_acc, loss.item())
        z_emb_model.save_to_checkpoint(epoch, loss, valid_acc)
    z_emb_model.get_test_accuracy()
    # (x, e) -> g -> y
    args['embs'] = 'g'
    train_dir = get_train_dir(wkdir, args, 'emb_net')
    writer = SummaryWriter(train_dir + 'logs/')
    g_emb_model = EmbeddingModel(args, wkdir, writer, True)
    start_epoch = g_emb_model.load_from_checkpoint(mode='latest')
    valid_acc = g_emb_model.get_test_accuracy(return_acc=True)
    for epoch in range(start_epoch, 5):
        # train one epoch
        loss = g_emb_model.train_one_epoch(epoch)
        # get validation accuracy
        valid_acc = g_emb_model.get_test_accuracy(return_acc=True)
        print(f'loss: {loss}')
        # save logs to json
        save_to_logs(train_dir, valid_acc, loss.item())
        # save model to checkpoint
        g_emb_model.save_to_checkpoint(epoch, loss, valid_acc)
    # get test accuracy
    g_emb_model.get_test_accuracy()
    


if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'nih', 'Dataset')
    flags.DEFINE_string('model', 'resnet18', 'Type of base model')
    flags.DEFINE_integer('num_classes', 9, 'Number of classes')
    flags.DEFINE_integer('batch', 128, 'Batchsize')
    flags.DEFINE_float('lr', 0.001, 'Learning rate')
    app.run(main)
