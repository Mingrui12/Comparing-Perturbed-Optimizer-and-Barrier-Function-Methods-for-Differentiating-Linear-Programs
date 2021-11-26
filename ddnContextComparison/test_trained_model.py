import torch
import torchvision
import torchvision.transforms as transforms
import Network
import projections

method = projections.Rand_Linprog
trained_model = torch.load('/students/u6392056/Server_Tasks/barrier_model_A_neg_t05/barrier_model_schemeA_batch32_t05_epoch4.pth')
for param_tensor in trained_model:
    print(param_tensor, "\t", trained_model[param_tensor].size())
net = Network.Net_schemeA_barrier()
net.load_state_dict(trained_model)
net.eval()

batch_size = 32

transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images)
correct = 0
total = 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('scheme A 60 epoch')
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
       
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))

