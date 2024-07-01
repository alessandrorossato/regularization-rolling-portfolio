import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV

def get_EF(returns):
    # Get the Covariance Matrix
    rCov = returns.cov()

    # A Column vector k by 1.
    vones = np.ones((rCov.shape[0],1))

    # The expected returns
    eR = returns.mean()
    eR = eR.to_numpy()
    eR = eR.reshape((eR.size, 1))

    # The standard deviation
    Rstd = returns.std()
    Rstd = Rstd.to_numpy()
    Rstd = Rstd.reshape((eR.size, 1))

    # Get A, B, C and delta
    A = vones.T@np.linalg.inv(rCov)@vones
    B = eR.T@np.linalg.inv(rCov)@vones
    C = eR.T@np.linalg.inv(rCov)@eR
    delta = A*C-B**2

    # Get the Equation of the EF
    c = C/delta
    fo = -2*B/delta
    so = A/delta

    EFret = np.linspace(-0.02,0.03,100)
    EFret = EFret.reshape((EFret.size, 1))
    EFvar = c + fo*EFret + so*EFret**2
    EFstd = np.sqrt(EFvar)

    return EFret, EFstd, eR, Rstd

def get_Xy(returns_stocks):
    eR = returns_stocks.mean() # Get the expected returns
    rCov = returns_stocks.cov() # Get the cov matrix

    R = returns_stocks - eR
    y = R.iloc[:,-1:]
    X = R.iloc[:,:-1]
    X = X.sub(y.iloc[:,0],axis=0)
    
    return X, y, R, eR, rCov

def equally_sectors(eq_weights, sector_dict, i):
    eq_weights['sector'] = [sector_dict.get(stock) for stock in eq_weights.index]
    eq_weights = eq_weights.groupby('sector').sum()
    eq_weights.to_csv(f'Sectors_weights/{i}_eq_weights.csv')
    return eq_weights

def pfLinReg(X, y, eR, rCov, sector_dict, i):

    LinReg = LinearRegression(fit_intercept=False).fit(X, y)

    LinRegPtf = pd.DataFrame(
        [
            list(X.columns)+list(y.columns),
            list(-LinReg.coef_[0]) + list([1+np.sum(LinReg.coef_[0])]) # weights = - coeff and w_n = 1-sum(w_i)
        ]
    ).transpose().set_index(0)

    LinRegPtf.columns = ['weight']
    LinRegPtf.index.name = 'firm'   
    # assign each stock to a sector with the sector_dict
    LinRegPtf['sector'] = [sector_dict.get(stock) for stock in LinRegPtf.index]
    
    LinRegPtfStd = np.sqrt(LinRegPtf['weight'].T@rCov@LinRegPtf['weight'])
    LinRegPtfEr = LinRegPtf['weight'].T@eR  

    # calculate the weights of the sectors
    regPtfgrouped = LinRegPtf.groupby('sector').sum()
    
    LinRegPtf.to_csv(f'Stocks_weights/{i}_LinRegPtf.csv')
    regPtfgrouped.to_csv(f'Sectors_weights/{i}_ElasticNetRegPtf.csv')
        
    return LinRegPtf, LinRegPtfStd, LinRegPtfEr, LinReg


def pfElasticNet(X, y, eR, rCov, sector_dict, i):
    l1_value = 1/4
    ElasticNetRegCV = ElasticNetCV(l1_ratio=l1_value,fit_intercept=False,max_iter=5000, tol=1e-04).fit(X,y)

    ElasticNetReg = ElasticNet(alpha=ElasticNetRegCV.alpha_,l1_ratio=l1_value,fit_intercept=False,max_iter=5000, tol=1e-04).fit(X,y)

    # Create dataFrame with corresponding feature and its respective coefficients
    ElasticNetRegPtf = pd.DataFrame(
        [
            list(X.columns)+list(y.columns),
            list(-ElasticNetReg.coef_) + list([1+np.sum(ElasticNetReg.coef_)]) # weights = - coeff and w_n = 1-sum(w_i)
        ]
    ).transpose().set_index(0)

    ElasticNetRegPtf.columns = ['weight']
    ElasticNetRegPtf.index.name = 'firm'
    # assign each stock to a sector with the sector_dict
    ElasticNetRegPtf['sector'] = [sector_dict.get(stock) for stock in ElasticNetRegPtf.index]

    ElasticNetRegPtfStd = np.sqrt(ElasticNetRegPtf['weight'].T@rCov@ElasticNetRegPtf['weight'])
    ElasticNetRegPtfEr = ElasticNetRegPtf['weight'].T@eR
    
    # calculate the weights of the sectors
    regPtfgrouped = ElasticNetRegPtf.groupby('sector').sum()
    
    ElasticNetRegPtf.to_csv(f'Stocks_weights/{i}_ElasticNetRegPtf.csv')
    regPtfgrouped.to_csv(f'Sectors_weights/{i}_ElasticNetRegPtf.csv')
    
    return ElasticNetRegPtf, ElasticNetRegPtfStd, ElasticNetRegPtfEr, ElasticNetReg