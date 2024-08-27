import numpy as np
import pandas as pd
import jax.numpy as jnp


def format_data(df, species, metabolites, controls, observed, batch_size=64):
    '''
    df is a dataframe with columns
    ['Experiments', 'Time', 'S_1', ..., 'S_ns', 'M_1', ..., 'M_nm', 'U_1', ..., 'U_nu']

    species := 'S_1', ..., 'S_ns'
    metabolites := 'M_1', ..., 'M_nm'
    controls := 'U_1', ..., 'U_nu'

    Format data into sets each with at most batch_size number of samples
     -Each set has the same number of evaluated time steps
    '''
    # concatenate all sytem variable names
    sys_vars = np.concatenate((species, metabolites))

    # divide dataframes by experiment name, keeping track of number of measurements in each exp
    groups = df.groupby("Experiments")

    # store dataframes with same number of measurements in a dictionary
    df_dict = {}
    for name, group in groups:
        if group.shape[0] not in df_dict.keys():
            df_dict[group.shape[0]] = [group]
        else:
            df_dict[group.shape[0]].append(group)

    # data is a list of tuples (T, X, U, Y, names) where each tuple corresponds to a batch
    # with the same measurement times
    data = []
    for n_eval in df_dict.keys():

        # pull samples with n_eval measurements
        df_n_eval = df_dict[n_eval]

        # number of samples with n_eval measurements
        n_samples = len(df_n_eval)

        # divide data into batches
        k = 0
        for batch_inds in np.array_split(np.arange(n_samples), np.ceil(n_samples / batch_size)):

            # initialize data matrix with NaNs
            T = np.empty([len(batch_inds), n_eval])
            T[:] = np.nan
            X = np.empty([len(batch_inds), len(sys_vars)])
            X[:] = np.nan
            U = np.empty([len(batch_inds), n_eval, len(controls)])
            U[:] = np.nan
            Y = np.empty([len(batch_inds), n_eval, len(observed)])
            Y[:] = np.nan

            # keep track of experiment names
            names = []
            for i, batch_ind in enumerate(batch_inds):

                # pull microbial community data
                comm_data = df_n_eval[k]
                k += 1

                # keep track of experiment names
                names.append(comm_data.Experiments.values[0])

                # store evaluation time points
                T[i] = np.array(comm_data.Time.values, float)

                # store initial condition data
                X[i] = np.array(comm_data[sys_vars].values, float)[0]

                # store controls and observed data
                U[i] = np.array(comm_data[controls].values, float)
                Y[i] = np.array(comm_data[observed].values, float)

                # force measurements of species that were not inoculated to be zero
                Y[i, :, :len(species)] = np.einsum('k,tk->tk', np.array(X[i, :len(species)] > 0, int), Y[i, :, :len(species)])

            data.append((T, X, U, Y, names))

    return data

# define scaling functions
class ZeroMaxScaler():

    def __init__(self, observed, system_variables):
        # store names of observed variables and system variables
        self.observed = observed
        self.system_variables = system_variables

    def fit(self, train_df):

        # save dictionary with {eval_time: max values}
        self.scale_dict_obs = {}
        self.scale_dict_sys = {}

        # scale by max values at each time slot
        time_groups = train_df.groupby("Time")
        for eval_time, eval_df in time_groups:
            max_vals = eval_df[self.observed].max().values
            max_vals[max_vals == 0] = 1.
            self.scale_dict_obs[eval_time] = max_vals

            max_vals = eval_df[self.system_variables].max().values
            max_vals[max_vals == 0] = 1.
            self.scale_dict_sys[eval_time] = max_vals
        return self

    def transform(self, test_df):
        # convert to 0-1 scale
        for eval_time in self.scale_dict_sys.keys():
            test_df.loc[test_df.Time.values == eval_time, self.system_variables] /= self.scale_dict_sys[eval_time]
        return test_df

    def inverse_transform(self, data):
        inv_data = []
        for T, pred, stdv, exps in data:
            for eval_time in self.scale_dict_obs.keys():
                t_inds = T == eval_time
                pred[t_inds] *= self.scale_dict_obs[eval_time]
                stdv[t_inds] *= self.scale_dict_obs[eval_time]
            inv_data.append((T, pred, stdv, exps))
        return inv_data

# define scaling functions
class MinMaxScaler():

    def __init__(self, observed, system_variables):
        # store names of observed variables and system variables
        self.observed = observed
        self.system_variables = system_variables

    def fit(self, train_df):

        # save dictionary with {eval_time: max values}
        self.eval_times  = np.unique(train_df.Time.values)
        self.scale_dict_obs = {}
        self.scale_dict_sys = {}

        # scale by max values at each time slot
        time_groups = train_df.groupby("Time")
        for eval_time, eval_df in time_groups:
            min_vals = eval_df[self.observed].min().values
            max_vals = eval_df[self.observed].max().values
            max_vals[max_vals == 0] = 1.
            min_vals[min_vals == max_vals] = 0.
            self.scale_dict_obs[f"{eval_time} min"] = np.clip(min_vals, 0, np.inf)
            self.scale_dict_obs[f"{eval_time} max"] = max_vals

            min_vals = eval_df[self.system_variables].min().values
            max_vals = eval_df[self.system_variables].max().values
            max_vals[max_vals == 0] = 1.
            min_vals[min_vals == max_vals] = 0.
            self.scale_dict_sys[f"{eval_time} min"] = np.clip(min_vals, 0, np.inf)
            self.scale_dict_sys[f"{eval_time} max"] = max_vals
        return self

    def transform(self, test_df):
        # convert to 0-1 scale
        for eval_time in self.eval_times:
            unscaled = test_df.loc[test_df.Time.values == eval_time, self.system_variables].values
            scaled = (unscaled - self.scale_dict_sys[f"{eval_time} min"]) / (self.scale_dict_sys[f"{eval_time} max"] - self.scale_dict_sys[f"{eval_time} min"])
            test_df.loc[test_df.Time.values == eval_time, self.system_variables] = scaled
        return test_df

    def inverse_transform(self, data):
        inv_data = []
        for T, pred, stdv, exps in data:
            for eval_time in self.eval_times:
                t_inds = T == eval_time
                pred[t_inds] *= (self.scale_dict_obs[f"{eval_time} max"] - self.scale_dict_obs[f"{eval_time} min"])
                pred[t_inds] += self.scale_dict_obs[f"{eval_time} min"]
                stdv[t_inds] *= (self.scale_dict_obs[f"{eval_time} max"] - self.scale_dict_obs[f"{eval_time} min"])
            inv_data.append((T, pred, stdv, exps))
        return inv_data
