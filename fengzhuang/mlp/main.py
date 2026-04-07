import function
## input ##
hiddenlayers = [10, 10, 10]
batch_size = 16
num_epochs = 20
lr  =0.0001
## load data ##
features= function.read_csv('feature_data.csv')
labels = function.read_csv('label_data.csv')
test = function.read_csv('test.csv')
net_structure = [len(features[0]), *hiddenlayers, len(labels[0])]
net = function.get_net(net_structure)

function.train(net, features, labels, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
outcome = function.use_net(net, test)
print(outcome)
