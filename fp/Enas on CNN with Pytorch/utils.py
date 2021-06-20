import torch


def layer_sampler(x):
    ##
    n = x.shape[0]
    rand = torch.rand(n, 1, 1)
    prob = torch.cumsum(x, dim=2)
    ## first layer shouldn't be pooling layer
    z = prob[0, 0, 2]
    # print('this is z', z)############################################################################################################################delete this
    rand[0, 0, 0] = rand[0, 0, 0] * (1 - z) + z  # to make sure 0 (max-pooling) and 1 (average-pooling) will not select.
    ##
    selected = torch.searchsorted(prob, rand)
    # print('this is selected', selected)############################################################################################################################delete this
    # # take log after sampling (we couldn't do it before sampling)
    # x = torch.log(x)
    #
    target = torch.Tensor(x.shape)
    for i in range(n):
        # one-hot vector of selected layer type
        if i % 2 == 0:
            s = torch.zeros(1, x.shape[2])
            # print('the shape of s is ', s.shape)############################################################################################################################delete this
            s[0, selected[i, 0, 0]] = 1
            target[i] = s
        else:
            target[i] = 0
            # target[i] = x[i]
    # print(target)
    return target


def connection_sampler(x):
    #     print(x.shape)
    #     print(x)
    target = torch.Tensor(x.shape)
    n = x.shape[0]
    for i in range(n):
        with torch.no_grad():
            target[i] = x[i]
        if i % 2 == 1:
            no_possible_connections = 0
            cont_status = True
            while cont_status:
                no_possible_connections = int((i + 1) / 2)
                s = torch.log(torch.rand(1, no_possible_connections))
                z = torch.zeros(1, no_possible_connections)
                z[s < target[i, 0][:no_possible_connections]] = 1
                cont_status = bool(torch.sum(z) == 0)
            target[i, 0][:no_possible_connections] = z
            target[i, 0][no_possible_connections:] = 0
        else:
            target[i] = 0

    # print('connection sampler' , target)
    return target


def moving_average(accuracy_list):
    return sum(accuracy_list) / len(accuracy_list)


def weighted_moving_average(accuracy_list):
    l = len(accuracy_list)
    numerator = 0
    denominator = 0
    for i in range(l):
        denominator += i
        numerator += accuracy_list[i]
    return numerator / denominator


def exponential_moving_average(accuracy_list, alpha=0.1):
    s = accuracy_list[0]
    l = len(accuracy_list)
    for i in range(1, l):
        s = alpha * accuracy_list[i] + (1 - alpha) * s
    return s
