import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)
cmap = ListedColormap(colors)
save_figures = True

def plot_weights(gen_weights,mus):
          
          ''' plots weights from an LM-HMM or GLM-HMM'''
          num_states=len(gen_weights)
          obs_dim=len(mus[0])
          input_dim=len(gen_weights[0][0])
          # fig = plt.figure(figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
          fig = plt.figure(figsize=(obs_dim*4, 5), dpi=80, facecolor='w', edgecolor='k')
          for iObs in range(obs_dim):
                    plt.subplot(1, obs_dim, iObs+1)
                    # cols = ['#ff7f00', '#4daf4a', '#377eb8']
                    for k in range(num_states):
                              plt.plot(range(input_dim+1), np.append(gen_weights[k][iObs],mus[k][iObs]), marker='o',
                                        color=colors[k], linestyle='-',
                                        lw=1.5, label="state " + str(k+1))
                    plt.yticks(fontsize=10)
                    plt.xlabel("covariate", fontsize=15)
                    plt.xticks(np.arange(input_dim+1),np.append(["input-"+str(i) for i in range(input_dim)],'bias'), fontsize=12, rotation=45)
                    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
                    plt.title("Weights for obs dim "+str(iObs), fontsize = 15)
                    if iObs==0: plt.legend(); plt.ylabel("LM weight", fontsize=15)
                    plt.tight_layout()
                    
def plot_weights_comparison(weight_dic):
          
          num_comp=len(weight_dic)
          ''' plots weights from an LM-HMM or GLM-HMM'''
          num_states=len(weight_dic[0]['weights'])
          obs_dim=len(weight_dic[0]['mus'][0])
          input_dim=len(weight_dic[0]['weights'][0][0])
          style_set=['-','--',':']
          marker_set=['o','s','d'];
          # fig = plt.figure(figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
          fig = plt.figure(figsize=(obs_dim*4, 5), dpi=80, facecolor='w', edgecolor='k')
          for iObs in range(obs_dim):
                    plt.subplot(1, obs_dim, iObs+1)
                    # cols = ['#ff7f00', '#4daf4a', '#377eb8']
                    for iComp in range(num_comp):
                              for k in range(num_states):
                                        plt.plot(range(input_dim+1), np.append(weight_dic[iComp]['weights'][k][iObs],
                                                  weight_dic[iComp]['mus'][k][iObs]), marker=marker_set[iComp],
                                                  color=colors[k], linestyle=style_set[iComp],
                                                  lw=1.5, label=(weight_dic[iComp]['label'])*(k==0))                                        
                    plt.yticks(fontsize=10)
                    plt.xlabel("covariate", fontsize=15)
                    plt.ylabel("weights", fontsize=15)
                    plt.xticks(np.arange(input_dim+1),np.append(["input-"+str(i) for i in range(input_dim)],'bias'), fontsize=12, rotation=45)
                    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
                    plt.title("Weights for obs dim "+str(iObs), fontsize = 15)
                    if iObs==0: plt.legend(); plt.ylabel("LM weight", fontsize=15)
                    plt.tight_layout()                    
                    
def plot_postprob_obs(posterior_probs0,data0,inputdata0,fit_slds,colors,cmap,XLIM=None):
          ''' '''
          fig, (a0, a01, a1) = plt.subplots(3,1, gridspec_kw={'height_ratios': [1,1,3]})

          # Plot the data and the smoothed data
          inpt=inputdata0; obs=data0; time_bins=len(inpt); obs_dim=len(obs[0])
          num_states=fit_slds.transitions.log_Ps.shape[0]
          input_dim=len(inputdata0[0]);

          for k in range(num_states):
                    a0.plot(posterior_probs0[:, k], label="State " + str(k + 1), lw=2,
                              color=colors[k])
          a0.set_ylim((-0.05, 1.05))
          a0.set_yticks([0, 1])
          a0.tick_params(axis='y', labelsize=15)
          a0.set_ylabel("p(state)", fontsize = 15)
          a0.set_xlim(0, time_bins)
          if XLIM is not None: a0.set_xlim(XLIM)
          a0.set_xticks([])
          
          lim_input = 1.1 * abs(inpt).max()
          for d in range(input_dim):
                    # a1.imshow(ind_state[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1, extent=(0, time_bins, -lim*obs_dim, (obs_dim)*lim), alpha=0.5)
                    a01.plot(inpt[:,d]- lim_input * d, '-c',label='inpt dim '+str(d))
          a01.set_xticks([])
          a01.set_xlim(0, time_bins)
          a01.set_yticks(-np.arange(input_dim) * lim_input, ["$x_{{ {} }}$".format(n+1) for n in range(input_dim)])
          a01.set_title('Input')
          if XLIM is not None: a01.set_xlim(XLIM)

          
          lim = 2 * abs(obs).max(); 
          state_detected=posterior_probs0
          ind_state=np.argmax(state_detected, axis = 1)
          indnot=np.all(state_detected<0.8,axis=1)
          ind_state[indnot]=-1
          cmap.set_under('w')

          Ey = fit_slds.observations.mus[ind_state]
          EW = fit_slds.observations.Wks[ind_state]
          for d in range(obs_dim):
                    a1.imshow(ind_state[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1, extent=(0, time_bins, -lim*obs_dim, lim), alpha=0.5)
                    a1.plot(obs[:,d]- lim * d, '-k',label='obs'*(d==0))
                    a1.plot(Ey[:,d]- lim * d, ':k',label='bias'*(d==0))
                    a1.plot(EW[:,d]- lim * d, '--k',label='weight'*(d==0))       
          a1.set_xlim(0, time_bins)
          a1.set_yticks(-np.arange(obs_dim) * lim, ["$y_{{ {} }}$".format(n+1) for n in range(obs_dim)])
          a1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
          a1.set_xlabel("Time steps")
          a1.set_ylabel("inputs,obs")
          if XLIM is not None: a1.set_xlim(XLIM) 
          # fig.tight_layout()
          return fig

def plot_trans_matrix(gen_trans_mat):
          plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
          num_states=gen_trans_mat.shape[0]
          for i in range(gen_trans_mat.shape[0]):
                    for j in range(gen_trans_mat.shape[1]):
                              text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
                                        color="k", fontsize=12)
          plt.xlim(-0.5, num_states - 0.5)
          plt.xticks(range(0, num_states), np.arange(num_states)+1, fontsize=10)
          plt.yticks(range(0, num_states), np.arange(num_states)+1, fontsize=10)
          plt.ylim(num_states - 0.5, -0.5)
          plt.ylabel("state t", fontsize = 15)
          plt.xlabel("state t+1", fontsize = 15)
          
          
def plot_cv_indices(cv, X, y, group, ax, n_splits,plotlabels={'class':'class','group':'group','x':'sample index'}, lw=10):
    # modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
    """Create a sample plot for indices of a cross-validation object."""

    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + [plotlabels['class'], plotlabels['group']]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel=plotlabels['x'],
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        # xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax