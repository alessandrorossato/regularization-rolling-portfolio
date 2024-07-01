import matplotlib.pyplot as plt

# def plot_ef(eR_ind, Rstd_ind, EFret, EFstd, eR, Rstd, LinRegPtfStd, LinRegPtfEr, LassoRegPtfStd, LassoRegPtfEr, ElasticNetRegPtfStd, ElasticNetRegPtfEr, risk_free_train, i):
def plot_ef(eR_ind, Rstd_ind, EFret, EFstd, eR, Rstd, LinRegPtfStd, LinRegPtfEr, ElasticNetRegPtfStd, ElasticNetRegPtfEr, risk_free_train, i):
    # Plot the Efficient Frontier
    plt.figure(figsize=(15,7))
    plt.plot(EFstd,EFret,'r-', label = 'EF')
    plt.plot(0, risk_free_train/252, 'bo', label='risk free rate')
    plt.scatter(LinRegPtfStd,LinRegPtfEr,facecolors='none', edgecolors='b', label = 'LinReg Ptf = GMV')
    # plt.scatter(LassoRegPtfStd,LassoRegPtfEr,facecolors='none', edgecolors='m', label = 'LassoReg Ptf')
    plt.plot(Rstd_ind, eR_ind, 'ro', label='index')
    plt.scatter(ElasticNetRegPtfStd,ElasticNetRegPtfEr,facecolors='none', edgecolors='c', label = 'ElasticNetReg Ptf')
    plt.title('Efficient Frontier')
    plt.xlabel('sigma')
    plt.ylabel('return')
    plt.axis('tight')
    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.grid(True)
    plt.savefig(f'EF/{i}_EF.png')
    
def plot_sector_weights(regPtf, sector_dict, i, regr):    
    # assign each stock to a sector with the sector_dict
    regPtf['sector'] = [sector_dict.get(stock) for stock in regPtf.index]

    # calculate the weights of the sectors
    regPtfgrouped = regPtf.groupby('sector').sum()
    regPtfPlus = regPtfgrouped[regPtfgrouped['weight'] > 0]

    # calculate the weights of the sectors
    regPtfMinus = regPtfgrouped[regPtfgrouped['weight'] < 0]

    # save a bar plot with positive and negative weights
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(regPtfPlus.index, regPtfPlus['weight'], label='positive')
    ax.bar(regPtfMinus.index, regPtfMinus['weight'], label='negative')
    plt.xticks(rotation=90)
    plt.title(f'{regr} Sector Weights')
    plt.xlabel('Sector')    
    plt.ylabel('Weight')
    plt.legend(loc='upper right')
    plt.savefig(f'Sectors/{i}_{regr}_sector_weights.png')
    
    
# def get_cum_returns(retW_lin, retW_lasso, retW_elastic, returns_index_test, i):	
def get_cum_returns(retW_eq, retW_lin, retW_elastic, returns_index_test, i):	
    # plot the returns of the portfolio and the index
    plt.figure(figsize=(15,7))
    plt.plot(retW_eq.cumsum(), label='equal weights')
    plt.plot(retW_lin.cumsum(), label='linear regression')
    plt.plot(retW_elastic.cumsum(), label='elastic net')
    plt.plot(returns_index_test.cumsum(), label='index')
    plt.title('Portfolios vs Index')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(loc='upper left')
    plt.savefig(f'Cumulative/{i}_cumulative.png')

