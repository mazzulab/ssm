import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.stats import norm
from autograd.misc.optimizers import sgd, adam
from ssm.optimizers import adamc, lbfgs
from autograd import grad
from autograd.numpy.numpy_boxes import ArrayBox

from ssm.util import ensure_args_are_lists


class _Hierarchical(object):
    """
    Base class for hierarchical models.  Maintains a parent class and a
    bunch of children with their own perturbed parameters.
    """
    def __init__(self, base_class, *args, tags=(None,), lmbda=0.01, **kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda

        # Top-level parameters (parent)
        self.parent = base_class(*args, **kwargs)
        self._sqrt_lmbdas = self.set_lmbdas()

        # Make models for each tag
        self.tags = tags
        self.children = dict()
        for tag in tags:
            self.children[tag] = base_class(*args, **kwargs)
            # ch = self.children[tag] = base_class(*args, **kwargs)
            # ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)
    
    def set_lmbdas(self):
        # Iterate over the list and set every entry in each array equal to the value
        lmbdas = [0]*len(self.parent.params)#initialize the lmdas list to have the same number of entries as there are distribution parameters
        for i, arr in enumerate(self.parent.params):
            if isinstance(self.lmbda, (np.ndarray, list, tuple)):#if lmbda is a vector or list for each parameter distribution
                lmbdas[i] = self.lmbda[i]
            else:#if lmbda is a scalar
                lmbdas[i] = self.lmbda#np.full_like(arr, self.lmbda)
        
        return np.array(lmbdas)

    def stack_child_params(self):
        #collect the parameter arrays as stacked arrays for each parameter type in temp
        temp = [[] for _ in range(len(self.parent.params))]
        num_children = len(self.children)
        for i, tag in enumerate(self.tags):
            for j in range(len(self.children[tag].params)):#for each param type in the tuple
                pvec = self.children[tag].params[j]
                #stack the parameter vector onto the lmdas vector
                if isinstance(pvec, ArrayBox):
                    pvec = pvec._value
                temp[j].append(pvec)

        stacked_params = [np.stack(temp[j],axis=-1) for j in range(len(temp))]
        return stacked_params

    def update_lmbdas(self):
        temp = self.stack_child_params()
        temp_lmbdas = [0]*len(self.parent.params)
        btwn_state_stds = [0]*len(self.parent.params)
        for j, spvecs in enumerate(temp):#for each parameter vector in the child params set
            #impute the lambda for each parameter as the standard deviation across all child params
            temp_lmbdas[j] = np.std(spvecs, axis=-1)
            #average across each parameter vector and then take the standard deviation
            btwn_state_stds[j] = np.std(np.mean(spvecs, axis=-1))
        
        #clip temp_lmbdas to be larger than 1E-6
        # temp_lmbdas = [np.mean(np.clip(lmbda, 1E-6, np.inf)) for lmbda in temp_lmbdas]
        temp_lmbdas = [np.clip(np.mean(lmbda), 7.5E-2, btwn_state_stds[j]) for j, lmbda in enumerate(temp_lmbdas)]

        # weights = np.exp(np.linspace(-1., 0., self.window_size))
        # weights /= weights.sum()
        # if len(self.lmda_memory) == self.window_size:
        #     temp_lmdas_mem = [0]*len(self.parent.params)
        #     for j in range(len(temp_lmdas)):#for each parameter vector in the child params set
        #         #moving average over all the stored lambdas in the window
        #         #collect the jth parameter vector across all child dists
        #         clams = [lmda_[j] for lmda_ in self.lmda_memory]
        #         temp_lmdas_mem[j] = np.average(clams, axis=-1)#, weights=weights)

        #     self.lmda_memory.pop(0)#remove the oldest set of parameter vector stds across child dists
        #     self.lmda_memory.append(temp_lmdas_mem)#add the new set of parameter vector stds across child dists
        #     new_lmdas = temp_lmdas_mem
        # else:
        #     self.lmda_memory.append(temp_lmdas)#save the standard deviations for the next iteration
        #     new_lmdas = temp_lmdas
        self.lmda_memory.append(temp_lmbdas)#save the standard deviations for the next iteration
        new_lmbdas = temp_lmbdas
        
        self._sqrt_lmbdas = np.array(new_lmbdas)
        # return tuple(temp_lmdas)
            
    @property
    def params(self):
        prms = (self.parent.params,)
        for tag in self.tags:
            prms += (self.children[tag].params,)
        if self.lambda_update == 'optimized':
            prms += (self._sqrt_lmbdas,)
        return prms

    @property
    def lambdas(self):
        return self._sqrt_lmbdas#**2

    @params.setter
    def params(self, value):
        self.parent.params = value[0]
        if self.lambda_update == 'optimized':
            for tag, prms in zip(self.tags, value[1:-1]):
                self.children[tag].params = prms
            self._sqrt_lmbdas = value[-1]
        else:
            for tag, prms in zip(self.tags, value[1:]):
                self.children[tag].params = prms

    def permute(self, perm):
        self.parent.permute(perm)
        for tag in self.tags:
            self.children[tag].permute(perm)
        if self.lambda_update == 'optimized':
            self._sqrt_lmbdas.permute(perm)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        #set the lmdas
        # self._sqrt_lmbdas = self.set_lmbdas()
        self.lmda_memory = []

        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags, init_method=init_method)
        # for tag in self.tags:
        #     if isinstance(self.lmbda, (np.ndarray, list, tuple)):
        #         self.children[tag].params = tuple(prm + np.sqrt(self.lmbda[i]) * npr.randn(*prm.shape) for i, prm in enumerate(self.parent.params))
        #     else:
        #         self.children[tag].params = tuple(prm + np.sqrt(self.lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)
            # self.children[tag].params = copy.deepcopy(self.parent.params)
        for tag in self.tags:
            self.children[tag].params = tuple(prm + self._sqrt_lmbdas[i] * npr.randn(*prm.shape) for i, prm in enumerate(self.parent.params))


    def log_prior(self):
        lmbdas = self.lambdas
        lp = self.parent.log_prior()
        # Gaussian likelihood on each child param given parent param
        for tag in self.tags:
            for pprm, cprm, cplmbda in zip(self.parent.params, self.children[tag].params, lmbdas):
                lp += np.sum(norm.logpdf(cprm, pprm, cplmbda))
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        # Optimize parent and child parameters at the same time with SGD
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):

            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):

                if hasattr(self.children[tag], 'log_initial_state_distn'):
                    log_pi0 = self.children[tag].log_initial_state_distn(data, input, mask, tag)
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(self.children[tag], 'log_transition_matrices'):
                    log_Ps = self.children[tag].log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)

                if hasattr(self.children[tag], 'log_likelihoods'):
                    lls = self.children[tag].log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, num_iters=num_iters, **kwargs)


class HierarchicalInitialStateDistribution(_Hierarchical):
    def log_initial_state_distn(self, data, input, mask, tag):
        return self.log_pi0 - logsumexp(self.log_pi0)


class HierarchicalTransitions(_Hierarchical):
    def log_transition_matrices(self, data, input, mask, tag):
        return self.children[tag].log_transition_matrices(data, input, mask, tag)


class HierarchicalObservations(_Hierarchical):
    def __init__(self, base_class, K, D, M, *args, tags=(None,), lmbda=0.01, gamma=0.8, lambda_update = 'recursive', **kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda
        self.window_size = 5
        self.gamma = gamma
        self.lambda_update = lambda_update
#         self.C = C
        self.M = M
        self.K = K
        self.D = D

        # Top-level parameters (parent)
        self.parent = base_class(K, D, M, *args, **kwargs)
        self._sqrt_lmbdas = self.set_lmbdas()

        # Make models for each tag
        self.tags = tags
        self.children = dict()
        for tag in tags:
            self.children[tag] = base_class(K, D, M, *args, **kwargs)
            # ch = self.children[tag] = base_class(K, D, M, *args, **kwargs)
            # ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

        # self.set_lmdas()
        # self.initialize()

    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag, with_noise=with_noise)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(expectations, data, input, tag)

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        # expected log joint
        def _expected_log_joint(expectations):
            if (self.lambda_update in ['recursive', 'optimized']):
                self.lmda_memory.append(self._sqrt_lmbdas)
            if self.lambda_update == 'recursive':
                self.update_lmbdas()

            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):

                if hasattr(self.children[tag], 'log_initial_state_distn'):
                    log_pi0 = self.children[tag].log_initial_state_distn(data, input, mask, tag)
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(self.children[tag], 'log_transition_matrices'):
                    log_Ps = self.children[tag].log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)

                if hasattr(self.children[tag], 'log_likelihoods'):
                    lls = self.children[tag].log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            if self.lambda_update == 'optimized':
                prior_param_pen = np.linalg.norm(np.array(self.params[-1]), ord=2)**2
            else:
                prior_param_pen = 0.
            obj = _expected_log_joint(expectations) - self.gamma*prior_param_pen
            return -obj / T

        # self.params = \
        #     optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)
        
        # self.params = \
        #     adamc(grad(_objective), self.params, num_iters=num_iters, **kwargs)

        lambda_lower_bound = 0.075#hiearchical prior lambda params cannot be negative
        # var_lower_bound = 1E-6#variance params cannot be negative
        #make bounds so that the None is returned for all params except the lambda params, the last entry in the params tuple
        low_bounds = []
        up_bounds = []
        param_count = len(self.params)
        if self.lambda_update == 'optimized':
            param_count -= 1
        
        if optimizer == 'lbfgs':
            for i in range(param_count):#for each param tuple.
                lb_tup = []
                ub_tup = []
                for j in range(len(self.params[i])):#for each param type in the tuple
                    # if j == len(self.params[i])-1:#the variance param is the last entry in the tuple
                    #     lb_tup.append(np.full_like(self.params[i][j], var_lower_bound))
                    #     ub_tup.append(np.full_like(self.params[i][j], np.inf))
                    # else:#all other params are bounded by +/- infinity
                    lb_tup.append(np.full_like(self.params[i][j], -np.inf))
                    ub_tup.append(np.full_like(self.params[i][j], np.inf))

                lb_tup = tuple(lb_tup)
                ub_tup = tuple(ub_tup)
                low_bounds.append(lb_tup)
                up_bounds.append(ub_tup)

            if self.lambda_update == 'optimized':
                #collect the parameter arrays as stacked arrays for each parameter type in temp
                temp = self.stack_child_params()
                #iterate over each parameter vector and calculate the standard deviation between the states 
                # of the average parameter vectors across all child distributions.
                btwn_state_stds = [0]*len(self.parent.params)
                for j, spvecs in enumerate(temp):#for each parameter vector in the child params set
                    #average across each parameter vector and then take the standard deviation
                    btwn_state_stds[j] = np.std(np.mean(spvecs, axis=-1))
                    # #ensure that the standard deviation is larger than the lower bound..
                    if btwn_state_stds[j] < lambda_lower_bound:
                        btwn_state_stds[j] = lambda_lower_bound + 1E-4
                #the last param is always the lambda params, one value for each observation model parameter
                low_bounds.append(tuple([np.full_like(self.params[-1][j], lambda_lower_bound) for j in range(len(self.params[-1]))]))#add the bounds for the lambda params
                up_bounds.append(tuple([np.full_like(self.params[-1][j], btwn_state_stds[j]) for j in range(len(self.params[-1]))]))#add the bounds for the lambda params, btwn_state_stds[j]
            #package into a single bounds tuple
            bounds = tuple(zip(low_bounds, up_bounds))
            #construct var_lower_bound array in the same shape as the variance params array
            self.params = \
                lbfgs(_objective, self.params, bounds=bounds, num_iters=num_iters, **kwargs)
        else:
            # Optimize parent and child parameters at the same time with SGD
            optimizer = dict(sgd=sgd, adam=adam, lbfgs=lbfgs)[optimizer]

            self.params = \
                optimizer(_objective, self.params, num_iters=num_iters, **kwargs)


class HierarchicalEmissions(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_y(self, z, x, input=None, tag=None):
        return self.children[tag].sample_y(z, x, input=input, tag=tag)

    def initialize_variational_params(self, data, input, mask, tag):
        return self.children[tag].initialize_variational_params(data, input, mask, tag)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        return self.children[tag].smooth(expected_states, variational_mean, data, input, mask, tag)

