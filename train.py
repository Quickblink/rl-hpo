import torch
import torch.nn.functional as F
from torch.distributions import Normal
from GPEnv import MultiEnv


def make_dataset_simple(num_batches, batch_size, max_iter, model, var, gamma, device):
    data = torch.empty((num_batches * batch_size, max_iter, 4), dtype=torch.float, requires_grad=False, device=device)
    env = MultiEnv(batch_size, max_iter, device)
    for i in range(num_batches):
        base = i * batch_size
        data[base:(base+batch_size), 0, :2] = env.reset()
        with torch.no_grad():
            out, v, hidden = model(data[base:(base+batch_size), 0, :2], None)
            action = Normal(out, var).sample()
            v_old = v.squeeze()
        for k in range(1, max_iter-1):
            data[base:(base+batch_size), k, :2], reward = env.step(action)
            with torch.no_grad():
                out, v, hidden = model(data[base:(base+batch_size), k, :2], hidden)
                action = Normal(out, var).sample()
                v = v.squeeze()
                vpr = gamma * v + reward
                data[base:base + batch_size, k, 2] = vpr
                data[base:base + batch_size, k, 3] = vpr - v_old
                v_old = v
        k = max_iter - 1
        data[base:(base + batch_size), k, :2], reward = env.step(action)
        data[base:base + batch_size, k, 2] = reward
        data[base:base + batch_size, k, 3] = reward - v_old
    return data


def make_dataset_multi(num_batches, batch_size, max_iter, model, var, gamma, device):
    data = torch.empty((2, num_batches * batch_size, max_iter, 3), dtype=torch.float, requires_grad=False, device=device)
    reward = torch.empty((2, batch_size), dtype=torch.float, requires_grad=False, device=device)
    env = MultiEnv(batch_size, max_iter*2, device)
    for i in range(num_batches):
        base = i * batch_size
        data[0, base:(base+batch_size), 0, :2] = env.reset()
        data[:, base:(base+batch_size), 0, 2] = torch.zeros((1), dtype=torch.float, requires_grad=False, device=device).expand(2, batch_size)
        with torch.no_grad():
            out, _, hidden = model(data[0, base:(base+batch_size), 0, :2], None)
            action = Normal(out, var).sample()
        for k in range(1, max_iter-1):
            data[1, base:(base+batch_size), k, :2], reward[1] = env.step(out, shadow=True)
            data[0, base:(base+batch_size), k, :2], reward[0] = env.step(action)
            with torch.no_grad():
                _, v1, _ = model(data[1, base:(base+batch_size), k, :2], hidden)
                out, v0, hidden = model(data[0, base:(base+batch_size), k, :2], hidden)
                action = Normal(out, var).sample()
                v0 = v0.squeeze()
                v1 = v1.squeeze()
                vpr0 = gamma * v0 + reward[0]  # .squeeze()
                vpr1 = gamma * v1 + reward[1]  # .squeeze()
                data[0, base:base + batch_size, k, 2] = vpr0 - vpr1
                data[1, base:base + batch_size, k, 2] = vpr1 #reward[1]
        k = max_iter - 1
        data[1, base:(base + batch_size), k, :2], reward[1] = env.step(out, shadow=True)
        data[0, base:(base + batch_size), k, :2], reward[0] = env.step(action)
        data[1, base:base + batch_size, k, 2] = reward[1]
        data[0, base:base + batch_size, k, 2] = reward[0] - reward[1]
    return data


#value+reward of all states, real and shadow
#compare real and shadow for advantage
#
def backward_batch_simple(batch0, model, max_iter, gamma, var, device):
    lossv = torch.zeros((1), dtype=torch.float, device=device)
    lossp = torch.zeros((1), dtype=torch.float, device=device)

    act0, v0, hidden = model(batch0[:, 0, :2], None)
    v_old = v0.squeeze()
    act_old = act0.squeeze()
    for k in range(1, max_iter-1):
        act0, v0, hidden = model(batch0[:, k, :2], hidden)
        act0 = act0.squeeze()
        v0 = v0.squeeze()
        lossv += F.mse_loss(v_old, batch0[:, k, 2])#.backward(retain_graph=True)
        adv = batch0[:, k, 3]
        loss = -adv * torch.exp(Normal(act_old, var).log_prob(batch0[:, k, 0]))
        lossp += loss.mean()#.backward(retain_graph=True)
        v_old = v0
        act_old = act0
    k = max_iter - 1
    lossv += F.mse_loss(v_old, batch0[:, k, 2])#.backward(retain_graph=True)
    adv = batch0[:, k, 3]
    loss = adv * torch.exp(Normal(act_old, var).log_prob(batch0[:, k, 0]))
    lossp += loss.mean()#.backward()
    #print('Loss:', lossv.item(), lossp.item())
    tloss = lossp + lossv
    tloss.backward()
    return lossv.item(), lossp.item()


#value+reward of all states, real and shadow
#compare real and shadow for advantage
#
def backward_batch(batch0, batch1, model, max_iter, gamma, var, device):
    lossv = torch.zeros((1), dtype=torch.float, device=device)
    lossp = torch.zeros((1), dtype=torch.float, device=device)

    act0, v0, hidden = model(batch0[:, 0, :2], None)
    v_old = v0.squeeze()
    act_old = act0.squeeze()
    for k in range(1, max_iter-1):
        # _, v1, _ = model(batch1[:, k, :2], hidden)
        act0, v0, hidden = model(batch0[:, k, :2], hidden)
        act0 = act0.squeeze()
        v0 = v0.squeeze()
        # v1 = v1.squeeze()
        vpr1 = batch1[:, k, 2]  # gamma * v1 +
        lossv += F.mse_loss(v_old, vpr1.detach())#.backward(retain_graph=True)
        adv = batch0[:, k, 2]
        loss = -adv * torch.exp(Normal(act_old, var).log_prob(batch0[:, k, 0]))
        lossp += loss.mean()#.backward(retain_graph=True)
        v_old = v0
        act_old = act0
    k = max_iter - 1
    lossv += F.mse_loss(v_old, batch1[:, k, 2])#.backward(retain_graph=True)
    adv = batch0[:, k, 2]
    loss = adv * torch.exp(Normal(act_old, var).log_prob(batch0[:, k, 0]))
    lossp += loss.mean()#.backward()
    #print('Loss:', lossv.item(), lossp.item())
    tloss = lossp + lossv
    tloss.backward()
    return lossv.item(), lossp.item()

def train(num_bigsteps, num_epochs, num_batches, batch_size, max_iter, model, var, gamma, opt, device):
    for bs in range(num_bigsteps):
        data = make_dataset_simple(num_batches, batch_size, max_iter, model, var, gamma, device)
        print('Bigstep: ', bs) #,', Avarage Advantage: ',data[:,:,3].sum()/data.shape[1]
        for e in range(num_epochs):
            idc = torch.randperm(data.shape[0], device=device)
            for i in range(num_batches):
                base = i*batch_size
                #batch0 = data[0, idc[base:base + batch_size]]
                #batch1 = data[1, idc[base:base + batch_size]]
                batch = data[idc[base:base + batch_size]]
                model.zero_grad()
                #lossv, lossp = backward_batch(batch0, batch1, model, max_iter, gamma, var, device)
                lossv, lossp = backward_batch_simple(batch, model, max_iter, gamma, var, device)
                opt.step()
                if i % 10 == 0:
                    print('Loss:', lossv, lossp)
            validate(1, batch_size, max_iter, model, device)


def validate(num_batches, batch_size, max_iter, model, device, render=False):
    env = MultiEnv(batch_size, max_iter, device)
    traj = []
    obs = env.reset()
    totr = 0
    hidden = None  # model.get_initial_state()
    for k in range(1, max_iter):
        with torch.no_grad():
            out, _, hidden = model(obs, hidden)
        obs, r = env.step(out)
        totr += r.sum()
        traj.append('%.2f' % out[0, 0].item())
    print('Policy Reward:', totr/(batch_size))
    print('Trajectory: ', traj)
    print('Last Action: ', out.squeeze()[:10])
    if render:
        env.render()
