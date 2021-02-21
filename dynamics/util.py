import torch as th
import numpy as np


def termination_fn_false(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.isfinite(next_obs).all(axis=-1) \
                * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                * (height > .7) \
                * (np.abs(angle) < .2)
    done = ~not_done
    done = done[:,None]
    return done

class FakeEnv:

    def __init__(self, model, env_id=None):
        self.model = model
        if env_id == 'Hopper-v2':
            self.termination_func = termination_fn_hopper
        elif env_id == 'HalfCheetah-v2':
            self.termination_func = termination_fn_false
        else:
            raise NotImplementedError

    def _get_logprob(self, x, means, variances):
        '''
            x : [ batch_size, obs_dim + 1 ]
            means : [ num_models, batch_size, obs_dim + 1 ]
            vars : [ num_models, batch_size, obs_dim + 1 ]
        '''
        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        #stds = np.std(means,0).mean(-1)
        #var_mean = np.var(means, axis=0, ddof=1).mean(axis=-1)
        maxes = np.max(np.linalg.norm(variances, axis=-1), axis=0)

        return log_prob, maxes

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        """ if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False """

        inputs = th.cat((obs, act), dim=-1).float().to(self.model.device)
        with th.no_grad():
            samples, ensemble_model_means, ensemble_model_logvars = self.model(inputs, deterministic=False, return_dist=True)
        obs = obs.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        #ensemble_model_means = ensemble_model_means.detach().cpu().numpy()
        #ensemble_model_logvars = ensemble_model_logvars.detach().cpu().numpy()
        #ensemble_model_vars = np.exp(ensemble_model_logvars)

        #ensemble_model_means[:,:,1:] += obs
        samples[:,:,1:] += obs
        #ensemble_model_stds = np.sqrt(ensemble_model_vars)

        #### choose one model from ensemble
        num_models, batch_size, _ = ensemble_model_means.shape
        model_inds = np.random.choice(self.model.elites, size=batch_size)
        batch_inds = np.arange(0, batch_size)
        samples = samples[model_inds, batch_inds]
        #model_means = ensemble_model_means[model_inds, batch_inds]
        #model_stds = ensemble_model_stds[model_inds, batch_inds]

        #log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.termination_func(obs, act, next_obs)

        #batch_size = model_means.shape[0]
        #return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        #return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        """ if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0] """

        #info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, {}
