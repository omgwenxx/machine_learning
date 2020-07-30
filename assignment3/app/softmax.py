import torch
import torch.nn.functional as F

class SoftmaxModel:
    """
    A class that wraps the softmax classification model
    See also https://gist.github.com/DuckSoft/8d251871aab29689aca23e41da133886
    """

    def __init__(self, features, classes):
        self.model = torch.nn.Linear(features, classes)  # a linear model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)  # stochastic gradient descent
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.features = features
        self.classes = classes

    def train(self, dataset, ntrain):
        loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
        iteration_count = 0  # count epochs, see paper page 9
        total_iterations = 0
        best = 0
        # we train so long as there is no improvement for 100 epochs
        while iteration_count < 100:
            total_iterations += 1
            for image, labels in loader:  # train batch of size batch_size for one epoch
                image = image.view(-1, self.features)  # get to right shape

                # forward
                output = self.model(image)  # predict
                loss = self.loss_function(output, labels)

                # backward
                self.optimizer.zero_grad()  # clear gradients of optimizer
                loss.backward()
                self.optimizer.step()

            # epoch over, count 1
            iteration_count += 1

            # compute accuracy on train set
            check_loader = torch.utils.data.DataLoader(dataset, batch_size=ntrain, shuffle=True)
            acc = self.evaluate(check_loader)
            if acc > best:  # if we improve our accuracy, set the iteration count to 0
                iteration_count = 0
                best = acc  # update best accuracy

        return total_iterations

    def test(self, dataset, ntest):
        loader = torch.utils.data.DataLoader(dataset, batch_size=ntest, shuffle=True)
        acc = self.evaluate(loader)
        return acc

    def evaluate(self, loader):
        with torch.no_grad():
            im, lab = iter(loader).next()
            samples = self.model(im.view(-1, self.features))
            pred = samples.argmax(dim=1)
            acc = (lab == pred).sum().item() / len(lab)
            # print("number of images:", len(lab))
            return acc

    def predict(self, sample):
        # softmax prediction
        return F.softmax(self.model(sample.view(1, self.features)), dim=1)
