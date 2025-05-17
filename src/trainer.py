import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, device, optimizer=None, criterion=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion if criterion else torch.nn.CrossEntropyLoss()

    def train(self, dataloader, epochs=1):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for _ in range(epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def get_gradients(self, dataloader):
        """
        Get gradients for one pass through the data.
        Used for FedSGD to send gradients to server.
        """
        self.model.train()
        self.optimizer.zero_grad()

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            # For FedSGD, only one batch is used; break here
            break

        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
        return grads

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += data.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
