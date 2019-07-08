import torch
from torch import nn, cuda, optim
from torch.utils.data import DataLoader

from classes import MNISTDataset, LSTM

import time
import argparse as ap


def main(args: ap.Namespace):

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    batch_print = 50

    t_stmp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    train_set = MNISTDataset(mode="TRAIN")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True
                              )

    test_set = MNISTDataset(mode="TRAIN")
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.test_batch,
                             shuffle=True
                             )

    model = LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr,
                           weight_decay=args.lamb
                           )

    running_loss = 0

    for epoch in range(args.epochs):

        print("")

        for i, (digit, label) in enumerate(train_loader):

            digit = digit.to(device)
            label = label.to(device)

            output = model(digit)

            loss = criterion(output, label)

            running_loss += loss.item()

            loss.backward()

            optimizer.step()

            if (i + 1) % batch_print == 0:

                print("\rStep: {0}\tLoss: {1:.3f}\tAccuracy: (2:.3f)".format(i,
                                                                             running_loss / batch_print,
                                                                             running_loss),
                      end="")

    pass


if __name__ == '__main__':

    parser = ap.ArgumentParser()

    parser.add_argument("-b",
                        "--batch-size",
                        type=int,
                        default=2,
                        help="Batch Size (default 2).")

    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=25,
                        help="Number of epochs (default 25).")

    parser.add_argument("--lr",
                        # "--learning-rate",
                        type=float,
                        default=0.01,
                        help="Initial learning rate (default 0.01)")

    parser.add_argument("--test-batch",
                        type=int,
                        default=2,
                        help="Number of batches for testing (default 2).")

    parser.add_argument("-l",
                        "--lamb",
                        type=float,
                        default=0,
                        help="L2 Norm (default 0)"
                        )
    #
    # parser.add_argument("-d",
    #                     "--lr-decay",
    #                     type=float,
    #                     default=0.95,
    #                     help="Learning rate decay (default 0.95)")
    #
    # parser.add_argument("-f",
    #                     "--features",
    #                     type=int,
    #                     default=1024,
    #                     help="Number of features for penultimate layer")
    #
    # parser.add_argument("-w",
    #                     "--file-write",
    #                     action="store_true",
    #                     default=False,
    #                     help="Write to file instead of stdout.")
    #
    # parser.add_argument("-c",
    #                     "--create-dummy",
    #                     action="store_true",
    #                     default=False,
    #                     help="Write a dummy model file for evaluation.")

    args_ = parser.parse_args()

    main(args_)
