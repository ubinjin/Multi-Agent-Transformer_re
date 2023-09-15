import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log

def discrete_autoregreesive_act_reverse(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    
    tensoR = torch.zeros(batch_size)
    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
        else:
            tensoR = action
    
    r_shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    r_shifted_action[:, 0, 0] = 1
    r_output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    r_output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    r_available_actions = torch.flip(available_actions, [1])
    r_obs = torch.flip(obs, [1])
    r_obs_rep = torch.flip(obs_rep, [1])
    for i in range(n_agent):
        if i == 0:
            r_output_action[:, i, :] = output_action[:, n_agent - 1, :]
            r_output_action_log[:, i, :] = output_action_log[:, n_agent - 1, :]
            r_action = tensoR
        else:
            logit = decoder(r_shifted_action, r_obs_rep, r_obs)[:, i, :]
            if r_available_actions is not None:
                logit[r_available_actions[:, i, :] == 0] = -1e10

            distri = Categorical(logits=logit)
            r_action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            r_action_log = distri.log_prob(r_action)

            r_output_action[:, i, :] = r_action.unsqueeze(-1)
            r_output_action_log[:, i, :] = r_action_log.unsqueeze(-1)
        
        #print("r_output_action: ", r_output_action)
        if i + 1 < n_agent:
            r_shifted_action[:, i + 1, 1:] = F.one_hot(r_action, num_classes=action_dim)
    output_action = torch.flip(r_output_action, [1]) 
    output_action_log = torch.flip(r_output_action_log, [1]) 
    return output_action, output_action_log

def discrete_autoregreesive_act_reverse_x(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False, iteration=0):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    
    tensoR = torch.zeros(batch_size)
    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
        else:
            tensoR = action
    
    r_shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    r_shifted_action[:, 0, 0] = 1
    r_output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    r_output_action_log = torch.zeros_like(r_output_action, dtype=torch.float32)
    r_available_actions = torch.flip(available_actions, [1])
    r_obs = torch.flip(obs, [1])
    r_obs_rep = torch.flip(obs_rep, [1])
    for i in range(n_agent):
        if i == 0:
            r_output_action[:, i, :] = output_action[:, n_agent - 1, :]
            r_output_action_log[:, i, :] = output_action_log[:, n_agent - 1, :]
            r_action = tensoR
        else:
            logit = decoder(r_shifted_action, r_obs_rep, r_obs)[:, i, :]
            if r_available_actions is not None:
                logit[r_available_actions[:, i, :] == 0] = -1e10

            distri = Categorical(logits=logit)
            r_action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            r_action_log = distri.log_prob(r_action)

            r_output_action[:, i, :] = r_action.unsqueeze(-1)
            r_output_action_log[:, i, :] = r_action_log.unsqueeze(-1)
        
        #print("r_output_action: ", r_output_action)
        if i + 1 < n_agent:
            r_shifted_action[:, i + 1, 1:] = F.one_hot(r_action, num_classes=action_dim)

    for j in range(iteration - 1):
        r_shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
        r_shifted_action[:, 0, 0] = 1
        r_available_actions = torch.flip(r_available_actions, [1])
        r_obs = torch.flip(r_obs, [1])
        r_obs_rep = torch.flip(r_obs_rep, [1])
        for i in range(n_agent):
            if i == 0:
                r_output_action[:, i, :] = r_output_action[:, n_agent - 1, :]
                r_output_action_log[:, i, :] = r_output_action_log[:, n_agent - 1, :]
            else:
                logit = decoder(r_shifted_action, r_obs_rep, r_obs)[:, i, :]
                if r_available_actions is not None:
                    logit[r_available_actions[:, i, :] == 0] = -1e10

                distri = Categorical(logits=logit)
                r_action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
                r_action_log = distri.log_prob(r_action)

                r_output_action[:, i, :] = r_action.unsqueeze(-1)
                r_output_action_log[:, i, :] = r_action_log.unsqueeze(-1)

            # print("r_output_action: ", r_output_action)
            if i + 1 < n_agent:
                r_shifted_action[:, i + 1, 1:] = F.one_hot(r_action, num_classes=action_dim)

    if iteration % 2 == 1:
        output_action = torch.flip(r_output_action, [1]) 
        output_action_log = torch.flip(r_output_action_log, [1]) 
    else:
        output_action = r_output_action
        output_action_log = r_output_action_log
    return output_action, output_action_log

def discrete_autoregreesive_act_sequence(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                          available_actions=None, deterministic=False):
    
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    for j in range(n_agent):
        if j == 0: shifted_action[:, 0, 0] = 1
        else: 
            shifted_action[:, 1:, :] = 0
            # print("j: ", j, "0: ", shifted_action[0, 0, :])
            # print("j: ", j, "1: ", shifted_action[0, 1, :])
            # print("j: ", j, "2: ", shifted_action[0, 2, :])
            # print("j: ", j, "3: ", shifted_action[0, 3, :])
            # print("j: ", j, "4: ", shifted_action[0, 4, :])
            # print("j: ", j, "5: ", shifted_action[0, 5, :])
        for i in range(n_agent):
            if i > 0 or j == 0:
                logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
                if available_actions is not None:
                    logit[available_actions[:, i, :] == 0] = -1e10

                distri = Categorical(logits=logit)
                action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
                action_log = distri.log_prob(action)

                output_action[:, i, :] = action.unsqueeze(-1)
                output_action_log[:, i, :] = action_log.unsqueeze(-1)
            # logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
            # if available_actions is not None:
            #     logit[available_actions[:, i, :] == 0] = -1e10

            # distri = Categorical(logits=logit)
            # action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            # action_log = distri.log_prob(action)

            # output_action[:, i, :] = action.unsqueeze(-1)
            # output_action_log[:, i, :] = action_log.unsqueeze(-1)
            if i + 1 < n_agent:
                shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
        
        output_action = torch.cat((output_action[:, -1:, :], output_action[:, :-1, :]), dim=1)
        output_action_log = torch.cat((output_action_log[:, -1:, :], output_action_log[:, :-1, :]), dim=1)
        available_actions = torch.cat((available_actions[:, -1:, :], available_actions[:, :-1, :]), dim=1)
        obs = torch.cat((obs[:, -1:, :], obs[:, :-1, :]), dim=1)
        obs_rep = torch.cat((obs_rep[:, -1:, :], obs_rep[:, :-1, :]), dim=1)
        shifted_action = torch.cat((shifted_action[:, -1:, :], shifted_action[:, :-1, :]), dim=1)

    return output_action, output_action_log

def discrete_parallel_act_sequence(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    # print("action_log: ", action_log)
    # print("entropy: ", entropy)
    return action_log, entropy

def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
