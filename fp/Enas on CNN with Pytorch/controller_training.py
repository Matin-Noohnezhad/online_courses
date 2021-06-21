import time

import torch
import torch.nn as nn
import torch.optim as optim

def train(controller_net, soft_1, soft_2, layers_types, skip_connections, policy_gradient_multiplier, grad_clip_value=0.25,
          no_epoch=1):
    n = 0
    optimizer = optim.Adam(controller_net.parameters(), lr=35e-5)

    ####################################### dirty codes #######################################
    # soft_1 = soft_1.reshape(soft_1.shape[0], soft_1.shape[2])
    # soft_2 = soft_2.reshape(soft_2.shape[0], soft_2.shape[2])
    # layers_types = layers_types.reshape(layers_types.shape[0], layers_types.shape[2])
    # skip_connections = skip_connections.reshape(skip_connections.shape[0], skip_connections.shape[2])
    # _, layers_types = torch.max(layers_types, 1)
    # _, skip_connections = torch.max(skip_connections, 1)
    ####################################### dirty codes #######################################
    start = time.time()
    # for epoch in range(1):
    for epoch in range(no_epoch):
        # net.to(self.device)
        controller_net.train()

        optimizer.zero_grad()

        soft_1, soft_2 = controller_net()
        soft_1 = torch.log(soft_1)

        reward1 = soft_1 * layers_types * policy_gradient_multiplier/100
        reward2 = soft_2 * skip_connections * policy_gradient_multiplier/100
        total_loss = - (reward1.mean() + reward2.mean())
        # print('matin noooooohnezhad')
        # total_loss.backward(torch.ones_like(total_loss) * policy_gradient_multiplier, retain_graph=True)
        # total_loss.backward(torch.ones_like(total_loss) * (policy_gradient_multiplier))
        total_loss.backward()
        nn.utils.clip_grad_norm_(controller_net.parameters(), max_norm=grad_clip_value)
        optimizer.step()
        #
        n += 1
        if (n % 500 == 0):
            end = time.time()
            print('step number ', n)
            print('The training time for last 100 epoch is: %.2f %% second' % (end - start))
            start = time.time()
    # return val_accuracy
