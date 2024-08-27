# import jax
from jax import jacrev
from functools import partial

# optimization libraries
from scipy.optimize import minimize

# stats
from jax.scipy.stats.norm import cdf, pdf

# matrix math
from .linalg import *

# progress bar
from tqdm import tqdm

# class that implements standard RNN
class miRNN():

    def __init__(self, n_species, n_metabolites, n_controls, n_hidden,
                 f_ind=1., param_0=1., rng_key=123):

        # set rng key
        rng_key = random.PRNGKey(rng_key)

        # store dimensions
        self.n_species = n_species
        self.n_metabolites = n_metabolites
        self.n_controls = n_controls
        self.n_hidden = n_hidden

        # spread of distribution to sample initial parameter values
        self.param_0 = param_0

        # fraction of independent samples (hyper-parameter)
        self.f_ind = f_ind

        # determine indeces of species, metabolites and controls
        self.n_out = n_species + n_metabolites
        self.s_inds = np.array([False] * self.n_out)
        self.m_inds = np.array([False] * self.n_out)
        self.s_inds[:n_species] = True
        self.m_inds[n_species:n_species + n_metabolites] = True

        # determine shapes of weights/biases = [Whh,bhh,Wih, Who,bho, h0]
        self.shapes = [[n_hidden, n_hidden], [n_hidden], [n_hidden, n_species + n_metabolites + n_controls],
                       [n_species + n_metabolites, n_hidden], [n_species + n_metabolites], [n_hidden]]
        self.k_params = []
        self.n_params = 0
        for shape in self.shapes:
            self.k_params.append(self.n_params)
            self.n_params += np.prod(shape)
        self.k_params.append(self.n_params)

        # initialize parameters
        self.params = np.zeros(self.n_params)
        for k1, k2, shape in zip(self.k_params[:-1], self.k_params[1:-1], self.shapes[:-1]):
            if len(shape) > 1:
                stdv = self.param_0 / np.sqrt(np.prod(shape))
            self.params[k1:k2] = random.uniform(rng_key, shape=(k2 - k1,), minval=0., maxval=stdv)
        self.Ainv = None
        self.a = 1e-4
        self.b = 1e-4

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, in_axes=(None, 0, 0)))

        # jit compile gradient w.r.t. params
        self.G = jit(jacfwd(self.forward_batch))
        self.Gi = jit(jacfwd(self.forward))

        # jit compile function to compute gradient of loss w.r.t. parameters
        self.compute_grad_NLL = jit(jacrev(self.compute_NLL))

    # reshape parameters into weight matrices and bias vectors
    @partial(jit, static_argnums=(0,))
    def reshape(self, params):
        # params is a vector = [Whh,bhh,Wih,Who,bho,h0]
        return [jnp.reshape(params[k1:k2], shape) for k1, k2, shape in
                zip(self.k_params, self.k_params[1:], self.shapes)]

    # per-sample prediction
    @partial(jit, static_argnums=(0,))
    def forward(self, params, sample, control):
        return self.output(params, sample[self.s_inds], sample[self.m_inds], control)

    @partial(jit, static_argnums=(0,))
    def output(self, params, s, m, u):
        # reshape params
        Whh, bhh, Wih, Who, bho, h0 = self.reshape(params)
        params = [Whh, bhh, Wih, Who, bho]

        # define rnn
        rnn_ctrl = partial(self.rnn_cell, params, u)

        # define initial value
        init = (0, h0, s, m)

        # per-example predictions
        carry, out = lax.scan(rnn_ctrl, init, xs=u[1:])
        return out

    # RNN cell
    def rnn_cell(self, params, u, carry, inp):
        # unpack carried values
        t, h, s, m = carry

        # params is a vector = [Whh,bhh,Wih,Who,bho]
        Whh, bhh, Wih, Who, bho = params

        # concatenate inputs
        i = jnp.concatenate((s, m, u[t]))

        # update hidden vector
        # h = nn.tanh(Whh @ h + Wih @ i + bhh)
        h = nn.leaky_relu(Whh @ h + Wih @ i + bhh)

        # predict output
        zeros_mask = jnp.concatenate((jnp.array(s > 0, jnp.float32), jnp.ones(m.shape)))
        o = zeros_mask * (Who @ h + bho)
        s, m = o[:len(s)], o[len(s):]

        # return carried values and slice of output
        return (t + 1, h, s, m), o

    def fit(self, data, alpha_0=0., alpha_1=1., evd_tol=1e-3, nlp_tol=None, patience=1, max_fails=5):
        # estimate parameters using gradient descent
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence = -np.inf

        # init convergence status
        converged = False

        # initialize hyper parameters
        self.init_hypers(data, alpha_0)

        while not converged:
            # update Alpha and Beta hyper-parameters
            if self.itr > 0:
                self.update_hypers(data)
                # nlp_tol = 1e-3

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian,
                                hess=self.hessian,
                                x0=self.params,
                                args=(data,),
                                tol=nlp_tol,
                                method='Newton-CG',
                                callback=self.callback)
            self.params = self.res.x
            self.loss = self.res.fun

            # update precision / covariance (Hessian / Hessian inverse)
            print("Updating precision...")
            self.update_precision(data)
            # update prior precision so that Hessian is positive definite
            if self.itr == 0:
                self.alpha = alpha_1 * np.ones_like(self.params)
            self.A, self.alpha = make_pos_def(self.A, self.alpha)
            self.update_covariance(data)

            # update evidence
            self.update_evidence()
            print("Evidence {:.3f}".format(self.evidence))

            # check convergence
            convergence = np.abs(previdence - self.evidence) / np.max([1., np.abs(self.evidence)])

            # update pass count
            if convergence < evd_tol:
                passes += 1
                print("Pass count ", passes)
            else:
                passes = 0

            # increment fails if convergence is negative
            if self.evidence < previdence:
                fails += 1
                print("Fail count ", fails)

            # determine whether algorithm has converged
            if passes >= patience:
                converged = True

            # terminate if maximum number of mis-steps exceeded
            if fails >= max_fails:
                print("Warning: Exceeded max number of attempts to increase model evidence, model could not converge.")
                converged = True

            # update evidence
            previdence = np.copy(self.evidence)
            self.itr += 1

    def callback(self, xk, res=None):
        print("Loss: {:.3f}, Residuals: {:.5f}".format(self.loss, self.res))
        return True

    # function to compute NLL loss function
    @partial(jit, static_argnums=(0,))
    def compute_NLL(self, params, X, U, Y, Beta):
        outputs = self.forward_batch(params, X, U)
        error = jnp.nan_to_num(outputs - Y[:, 1:])
        return jnp.einsum('ntk,kl,ntl->', error, Beta, error) / 2.

    # compute residuals
    @partial(jit, static_argnums=(0,))
    def compute_RES(self, params, X, U, Y, Beta):
        outputs = self.forward_batch(params, X, U)
        error = jnp.nan_to_num(outputs - Y[:, 1:])
        return jnp.mean(error)

    # define objective function
    def objective(self, params, data):
        # init loss with parameter penalty
        self.loss = jnp.dot(self.alpha * params, params) / 2.
        self.res = 0.

        # forward pass
        for (T, X, U, Y, _) in data:
            self.loss += self.compute_NLL(params, X, U, Y, self.Beta)
            self.res += self.compute_RES(params, X, U, Y, self.Beta) / len(data)

        return self.loss

    # define function to compute gradient of loss w.r.t. parameters
    def jacobian(self, params, data):

        # gradient of -log prior
        g = self.alpha * params

        # gradient of -log likelihood
        for (T, X, U, Y, _) in data:
            # backward
            g += self.compute_grad_NLL(params, X, U, Y, self.Beta)

        # return gradient of -log posterior
        return g

    # define function to compute approximate Hessian
    def hessian(self, params, data):
        # init w/ hessian of -log(prior)
        A = jnp.diag(self.alpha)

        # outer product approximation of hessian
        for (T, X, U, Y, _) in data:
            # Compute gradient of model output w.r.t. parameters
            G = self.G(params, X, U)

            # update Hessian
            A += A_next(G, self.Beta)

        return A

    # update hyperparameters alpha and Beta
    def init_hypers(self, data, alpha_0):
        # compute number of independent samples in the data
        self.N = jnp.zeros(self.n_out)
        yCOV = np.zeros(self.n_out)

        # for each batch of data
        for (T, X, U, Y, _) in data:
            # Y has shape [n_batch, n_t, n_out]

            # rough initial guess of measurement covariance (likely an over-estimate to start)
            yCOV += np.nansum(Y**2, (0, 1))

            # count the number of measured variables
            for Y_n in Y:
                # Y_n has shape [n_t, n_out]
                for Y_t in Y_n[1:]:
                    # count measurements if non-zero
                    self.N += np.array(Y_t > 0, int)

        # init alpha
        self.alpha = alpha_0 * jnp.ones_like(self.params)

        # divide by number of observations
        yCOV = yCOV / self.N

        # update beta
        self.Beta = jnp.diag(1./yCOV)
        self.BetaInv = jnp.diag(yCOV)

        print("Total measurements: {:.0f}, Number of parameters: {:.0f}, Initial regularization: {:.2e}".format(sum(self.N),
                                                                                                           self.n_params,
                                                                                                           alpha_0))

    # update hyper-parameters alpha and Beta
    def update_hypers(self, data):

        # compute measurement covariance
        yCOV = np.zeros(self.n_out)
        for (T, X, U, Y, _) in data:
            # forward, shape [n_batch, n_t, n_out]
            outputs = self.forward_batch(self.params, X, U)
            error = jnp.nan_to_num(outputs - Y[:, 1:])

            # compute gradients, shape [n_batch, n_t, n_out, n_params]
            G = self.G(self.params, X, U)

            # sum of measurement covariance update
            yCOV += np.sum(error**2, (0, 1)) + trace_GGM(G, self.Ainv)

        # update alpha
        self.alpha = 1. / (self.params ** 2 + jnp.diag(self.Ainv))
        # self.alpha = alpha*jnp.ones_like(self.params)
        # alpha = self.n_params / (jnp.sum(self.params**2) + jnp.trace(self.Ainv) + 2.*self.a)

        # divide by number of observations
        yCOV = yCOV / self.N

        # update beta
        self.Beta = jnp.diag(1./yCOV)
        self.BetaInv = jnp.diag(yCOV)

    # compute precision matrix
    def update_precision(self, data):

        # compute inverse precision (covariance Matrix)
        self.A = np.diag(self.alpha)
        for (T, X, U, Y, _) in data:
            # compute gradient of model w.r.t. parameters [n_batch, n_t, n_out, n_params]
            G = self.G(self.params, X, U)
            self.A += A_next(G, self.Beta)
            self.A = (self.A + self.A.T)/2.

    # compute covariance matrix
    def update_covariance(self, data):

        ### fast / approximate method: ###
        # self.Ainv, _ = make_pos_def(compute_Ainv(self.A), jnp.ones(self.n_params))

        # compute inverse precision (covariance Matrix)
        self.Ainv = np.diag(1./self.alpha)
        for (T, X, U, Y, _) in data:
            # update Ainv
            G = self.G(self.params, X, U)
            for Gn in G:
                for Gt in Gn:
                    self.Ainv -= Ainv_next(Gt, self.Ainv, self.BetaInv)

        # make sure that matrices are symmetric
        self.Ainv = (self.Ainv + self.Ainv.T)/2.

        # make sure Ainv is positive definite
        self.Ainv, _ = make_pos_def(self.Ainv, jnp.ones_like(self.alpha))

    # compute the log marginal likelihood
    def update_evidence(self):
        # compute evidence
        self.evidence = np.sum(self.N*np.log(np.diag(self.Beta)))/2. + \
                        np.nansum(np.log(self.alpha))/2. - \
                        log_det(self.A)/2. - self.loss

    # function to predict metabolites and variance
    def predict(self, data):

        # function to get diagonal of a tensor
        get_diag = vmap(vmap(jnp.diag, (0,)), (0,))

        # return batches of predictions
        predictions = []
        for T, X, U, Y, exp_names in data:
            # keep initial condition
            preds = np.array(self.predict_point(X, U))

            # compute sensitivities
            G = self.G(self.params, X, U)

            # compute covariances
            COV = np.array(compute_predCOV(self.BetaInv, G, self.Ainv))

            # pull out standard deviations
            stdvs = np.concatenate((np.zeros_like(np.expand_dims(X, 1)), np.sqrt(get_diag(COV))), 1)

            # append set
            predictions.append((T, preds, stdvs, exp_names))

        return predictions

    # function to predict metabolites and variance
    def predict_point(self, X, U):
        # make point predictions
        preds = nn.relu(self.forward_batch(self.params, X, U))

        # include known initial conditions
        preds = np.concatenate((np.expand_dims(X, 1), preds), 1)

        return preds

    # search for next best experiment
    def get_next_experiment(self, f_P, f_I, data, best_experiments, explore, max_explore):

        # init with previous selected experiment
        next_experiment = best_experiments[-1]
        w = np.copy(explore)
        while next_experiment in best_experiments and explore < max_explore:

            # evaluate utility of each experimental condition
            utilities = []
            max_utilities = []
            for f_P_i, f_I_i in zip(f_P, f_I):
                utility = f_P_i + w*f_I_i
                utilities.append(utility)
                max_utilities.append(np.max(utility))

            # select next best condition
            best_dim = np.argmax(max_utilities)
            best_sample = np.argmax(utilities[best_dim])
            T, X, U, exp_names = data[best_dim]
            next_experiment = exp_names[best_sample]

            # increase exploration rate
            w = w + explore

        return best_dim, best_sample, next_experiment, w

    # return indices of optimal samples using algorithm similar to upper confidence bound sampling
    def search_UCB(self, data, objective, n_design, explore=1e-3, max_explore=1e3, batch_size=512):

        # compute profit function (f: R^[n_t, n_o] -> R) in batches
        objective_batch = jit(vmap(objective))

        # for each data dimension
        print("Evaluating exploitation objective...")
        f_P = []
        f_P_max = []
        for T, X, U, exp_names in tqdm(data):

            # number of samples in each dimension
            n_samples = len(T)

            # make predictions on data
            preds = self.predict_point(X, U)
            f_P_i = objective_batch(preds)

            # append to list of objective evaluations
            f_P.append(f_P_i)
            f_P_max.append(np.max(f_P_i))

        # initialize with sample that maximizes objective
        best_dim = np.argmax(f_P_max)
        best_sample = np.argmax(f_P[best_dim])
        T, X, U, exp_names = data[best_dim]
        best_experiments = [exp_names[best_sample]]
        print("Picked experiment {}, with predicted outcome of {:.3f}".format(best_experiments[-1],
                                                                              f_P[best_dim][best_sample]))

        # init and update parameter covariance
        Ainv_q = jnp.copy(self.Ainv)
        Gi = self.Gi(self.params, X[best_sample], U[best_sample])
        for Gt in Gi:
            Ainv_q -= Ainv_next(Gt, Ainv_q, self.BetaInv)
        Ainv_q, _ = make_pos_def(Ainv_q, jnp.ones(self.n_params))

        # search for new experiments until find N
        while len(best_experiments) < n_design:

            # compute information acquisition function
            # for each data dimension
            print("Updating exploration utilities...")
            f_I = []
            f_I_max = []
            for T, X, U, exp_names in tqdm(data):
                # number of samples in each dimension
                n_samples = len(T)

                # compute sensitivities
                G = self.G(self.params, X, U)

                # compute covariances
                searchCOV = compute_searchCOV(self.BetaInv, G, Ainv_q)

                # make predictions on data
                f_I_i = batch_log_det(searchCOV)

                # append to list of objective evaluations
                f_I.append(f_I_i)
                f_I_max.append(np.max(f_I_i))

            # select next point
            best_dim, best_sample, next_experiment, w = self.get_next_experiment(f_P, f_I, data, best_experiments,
                                                                                 np.copy(explore), max_explore)
            best_experiments.append(next_experiment)
            print("Picked experiment {}, with exploration weight {:.5f}".format(best_experiments[-1], w))

            # update parameter covariance given selected condition
            T, X, U, exp_names = data[best_dim]
            Gi = self.Gi(self.params, X[best_sample], U[best_sample])
            for Gt in Gi:
                Ainv_q -= Ainv_next(Gt, Ainv_q, self.BetaInv)
            Ainv_q, _ = make_pos_def(Ainv_q, jnp.ones(self.n_params))

        # if have enough selected experiments, return
        return best_experiments

    # return indeces of optimally informative samples
    def explore(self, data, n_design):

        # init parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # search for new experiments until find N
        best_experiments = []
        while len(best_experiments) < n_design:

            # compute information acquisition function
            # for each data dimension
            f_I = []
            f_I_max = []
            for T, X, U, exp_names in tqdm(data):
                # compute sensitivities
                G = self.G(self.params, X, U)

                # compute covariances
                searchCOV = compute_searchCOV(self.BetaInv, G, Ainv_q)

                # make predictions on data
                f_I_i = batch_log_det(searchCOV)
                f_I.append(f_I_i)
                f_I_max.append(np.max(f_I_i))

            # select next point
            best_dim = np.argmax(f_I_max)
            best_sample = np.argmax(f_I[best_dim])
            T, X, U, exp_names = data[best_dim]

            # update parameter covariance given selected condition
            Gi = self.Gi(self.params, X[best_sample], U[best_sample])
            for Gt in Gi:
                Ainv_q -= Ainv_next(Gt, Ainv_q, self.BetaInv)
            Ainv_q, _ = make_pos_def(Ainv_q, jnp.ones(self.n_params))

            # only add to list if not already there
            if exp_names[best_sample] not in best_experiments:
                best_experiments.append(exp_names[best_sample])
                print("Picked experiment {}".format(best_experiments[-1]))
            else:
                print("Picked duplicate {}".format(exp_names[best_sample]))

        # if have enough selected experiments, return
        return best_experiments

    # return indices of optimal samples
    def exploit(self, data, objective, n_design, batch_size=512):

        # compute profit function (f: R^[n_t, n_o] -> R) in batches
        objective_batch = jit(vmap(objective))

        # for each data dimension
        f_P = []
        all_exp_names = []
        for T, X, U, Y, exp_names in data:
            # number of samples in each dimension
            n_samples = len(T)

            # compute objectives
            f_P_i = np.zeros(n_samples)
            for batch_inds in np.array_split(np.arange(n_samples), n_samples // batch_size):
                # make predictions on data
                preds = self.predict_point(X[batch_inds], U[batch_inds])
                f_P_i[batch_inds] = objective_batch(preds)

            # append to list of objective evaluations
            f_P.append(f_P_i)
            all_exp_names.append(exp_names)

        # flatten objective values
        f_P = np.concatenate(f_P)
        all_exp_names = np.concatenate(all_exp_names)

        # initialize with sample that maximizes objective
        best_inds = np.argsort(f_P)[::-1]

        # return list of top experiments
        return list(all_exp_names[best_inds[:n_design]])

# inherit class that implements microbiome RNN (miRNN)
class LR(miRNN):

    def __init__(self, n_species, n_metabolites, n_controls,
                 f_ind=1., param_0=1., rng_key=123):

        # set rng key
        rng_key = random.PRNGKey(rng_key)

        # store dimensions
        self.n_species = n_species
        self.n_metabolites = n_metabolites
        self.n_controls = n_controls
        self.param_0 = param_0
        self.f_ind = f_ind

        # determine indeces of species, metabolites and controls
        self.n_out = n_species + n_metabolites
        self.s_inds = np.array([False] * self.n_out)
        self.m_inds = np.array([False] * self.n_out)
        self.s_inds[:n_species] = True
        self.m_inds[n_species:n_species + n_metabolites] = True

        # determine shapes of weights/biases = [Whh,bhh,Wih, Who,bho, h0]
        self.shapes = [[n_species + n_metabolites, n_species + n_metabolites + n_controls], [n_species + n_metabolites]]
        self.k_params = []
        self.n_params = 0
        for shape in self.shapes:
            self.k_params.append(self.n_params)
            self.n_params += np.prod(shape)
        self.k_params.append(self.n_params)

        # initialize parameters
        self.params = np.zeros(self.n_params)
        for k1, k2, shape in zip(self.k_params[:-1], self.k_params[1:-1], self.shapes[:-1]):
            if len(shape) > 1:
                stdv = self.param_0 / np.sqrt(np.prod(shape))
            self.params[k1:k2] = random.uniform(rng_key, shape=(k2 - k1,), minval=0., maxval=stdv)
        self.Ainv = None
        self.a = 1e-4
        self.b = 1e-4

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, in_axes=(None, 0, 0)))

        # jit compile gradient w.r.t. params
        self.G = jit(jacfwd(self.forward_batch))
        self.Gi = jit(jacfwd(self.forward))

        # jit compile function to compute gradient of loss w.r.t. parameters
        self.compute_grad_NLL = jit(jacrev(self.compute_NLL))


    @partial(jit, static_argnums=(0,))
    def output(self, params, s, m, u):

        # define rnn
        rnn_ctrl = partial(self.rnn_cell, self.reshape(params), u)

        # define initial value
        init = (0, s, m)

        # per-example predictions
        carry, out = lax.scan(rnn_ctrl, init, xs=u[1:])
        return out

    # RNN cell
    @partial(jit, static_argnums=(0,))
    def rnn_cell(self, params, u, carry, inp):
        # unpack carried values
        t, s, m = carry

        # params is a vector = [Whh,bhh,Wih,Who,bho]
        A, b = params

        # concatenate inputs
        i = jnp.concatenate((s, m, u[t]))

        # predict output
        zeros_mask = jnp.concatenate((jnp.array(s > 0, jnp.float32), jnp.ones(m.shape)))
        o = zeros_mask*(A@i + b)
        # o = A@i + b
        # o = jnp.concatenate((s,m))*jnp.exp2(A@i + b)
        s, m = o[:len(s)], o[len(s):]

        # return carried values and slice of output
        return (t + 1, s, m), o
