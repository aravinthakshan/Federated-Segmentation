import argparse
import copy
import torch
import random
import numpy as np

from NN import FCN
from CNN import SimpleCNN
from dataloader import get_dataset, iid_split, non_iid_split, get_client_loaders, get_test_loader
from trainer import Trainer
from fedavg import fedavg
from fedsgd import fedsgd

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        set_seed(args.seed)

    # Load dataset
    trainset, testset = get_dataset(args.seed)

    # Split data IID or non-IID
    if args.iid:
        client_idxs = iid_split(trainset, args.num_clients, seed=args.seed)
    else:
        client_idxs = non_iid_split(trainset, args.num_clients, seed=args.seed)

    client_loaders = get_client_loaders(trainset, client_idxs, args.batch_size)
    test_loader = get_test_loader(testset, args.batch_size)

    # Initialize global model
    if args.model == 'cnn':
        global_model = SimpleCNN().to(device)
    else:
        global_model = FCN().to(device)

    # Federated training rounds
    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        local_models = []
        local_gradients = []

        for client_idx in range(args.num_clients):
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr)
            trainer = Trainer(local_model, device, optimizer)

            if args.algorithm == 'fedavg':
                # Train locally for several epochs
                loss, acc = trainer.train(client_loaders[client_idx], epochs=args.local_epochs)
                print(f" Client {client_idx} train loss: {loss:.4f}, acc: {acc:.4f}")
                local_models.append(local_model)

            elif args.algorithm == 'fedsgd':
                # Compute gradients on one batch (FedSGD style)
                grads = trainer.get_gradients(client_loaders[client_idx])
                local_gradients.append(grads)

        # Aggregate updates
        if args.algorithm == 'fedavg':
            global_model = fedavg(local_models).to(device)
        elif args.algorithm == 'fedsgd':
            global_model = fedsgd(global_model, local_gradients, lr=args.lr).to(device)

        # Evaluate global model
        eval_trainer = Trainer(global_model, device)
        test_loss, test_acc = eval_trainer.evaluate(test_loader)
        print(f" Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning CIFAR-10")

    parser.add_argument('--num_clients', type=int, default=5, help='number of clients')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'fcn'], help='model type')
    parser.add_argument('--iid', action='store_true', help='use IID data split')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--rounds', type=int, default=10, help='number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='local training epochs for FedAvg')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--algorithm', type=str, default='fedavg', choices=['fedavg', 'fedsgd'], help='federated algorithm')

    args = parser.parse_args()
    main(args)
