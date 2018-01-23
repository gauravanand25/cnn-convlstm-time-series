import argparse
import torch
import os.path

import ConvLSTMAE
import conv_auto_encoder
from conv_auto_encoder import ConvAutoEncoder

import classify_embeddings as svm_classify
from data_utils import MultipleDatasets, train_datasets, test_datasets, my_train_datasets, timenet_train_datasets, timenet_val_datasets
from utils import args_to_string, make_directory

def conv_lstm_ae(args, out_dir, str_args):

    ucr = MultipleDatasets(directory="./UCR_TS_Archive_2015", batch_size=args.batch_size,
                           datasets=timenet_train_datasets, merge_train_test=True, val_percentage=0)
    ucr.load_data()

    val_ucr = MultipleDatasets(directory="./UCR_TS_Archive_2015", batch_size=args.batch_size,
                           datasets=timenet_val_datasets, merge_train_test=True, val_percentage=0)
    val_ucr.load_data()

    conv_lstm_encoder, conv_lstm_decoder = ConvLSTMAE.fit(args, ucr, val_ucr)

    svm_classify.fit_lstm(args, conv_lstm_encoder, out_dir)

def conv_ae(args, out_dir, str_args):
    model_file = './results/best_trained_model.pt'
    train = not os.path.exists(model_file)
    if train:
        conv_model = ConvAutoEncoder(hidden_channels=[32, 16, 4], kernel_sizes=[20, 11, 8], strides=[2, 4, 2], padding=[0, 0, 0])
    else:
        conv_model = torch.load(model_file)

    conv_ucr = MultipleDatasets(directory="./UCR_TS_Archive_2015", batch_size=args.batch_size,
                                datasets=timenet_train_datasets, data_length=512, merge_train_test=True, val_percentage=0)
    conv_ucr.load_data()
    conv_collated_data = conv_ucr.collate()

    val_data = MultipleDatasets(directory="./UCR_TS_Archive_2015", batch_size=args.batch_size,
                                datasets=timenet_val_datasets, data_length=512, merge_train_test=True,
                                val_percentage=0)
    val_data.load_data()
    val_data = val_data.collate()

    conv_model.train()      #for dropout and batchnorm
    conv_auto_encoder.fit(args, conv_collated_data, val_data, conv_model)

    if train:  # save model
        torch.save(conv_model, out_dir + '/' + str_args + '.pt')

    conv_model.eval()       #for dropout and batchnorm
    svm_classify.fit_conv_ae(conv_model, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conv AE')
    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--model', default='ConvAE', type=str, metavar='N', help='Which model "ConvAE" or "ConvLSTMAE"?')
    parser.add_argument('--epochs', default=6, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--filter_size', default=9, type=int, metavar='N', help='kernel size')
    parser.add_argument('--conv_len', default=32, type=int, metavar='N', help='conv len in LSTM')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate')
    args = parser.parse_args()

    str_args = args_to_string(args)
    out_dir = './search/' + str_args
    make_directory(out_dir)
    print str_args

    if args.model == 'ConvAE':
        conv_ae(args, out_dir, str_args)
    elif args.model == 'ConvLSTMAE':
        conv_lstm_ae(args, out_dir, str_args)
