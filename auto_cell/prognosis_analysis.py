import os
import glob
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import lifelines
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from sklearn.preprocessing import StandardScaler
from os.path import join as join
import random
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression,Lasso,LogisticRegressionCV,LinearRegression,RidgeClassifier
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from utils import *

plt.rcParams['font.family'] = 'Arial'
def sklearn_vif(exogs, data):
    '''
    This function calculates variance inflation function in sklearn way.
     It is a comparatively faster process.
    '''
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}
    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]
        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)
        # calculate VIF
        vif = 1 / (1 - r_squared)
        vif_dict[exog] = vif
        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance
    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})
    return df_vif
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
def sig(x):
    return 1 / (1 + np.exp(-x))

rt_path = ''
feature_num = 66
csv_fin_path = join(rt_path, 'df_io_fin.csv')
cell_df = pd.read_csv(csv_fin_path)
cell_df.replace([np.inf, -np.inf], np.nan, inplace=True)
mask = cell_df['pfs'].isna()
cell_df = cell_df[~mask].copy()
cell_df = cell_df.reset_index(drop=True)
cell_df = cell_df.groupby('patient_id').mean()
cell_df['patient_id'] = list(cell_df.index)
cell_df = cell_df.drop_duplicates(subset=['patient_id'], keep='first')
cell_df = cell_df.reset_index(drop=True)
biomarkers = cell_df.iloc[:, :feature_num].copy()
patient_info = cell_df.iloc[:, feature_num:]
scaler = StandardScaler().fit(biomarkers)
scaler_biomarkers = pd.DataFrame(scaler.transform(biomarkers))
scaler_biomarkers.columns = biomarkers.columns
df = scaler_biomarkers.dropna().copy()
df_vif = sklearn_vif(exogs=df.columns, data=df).sort_values(by='VIF', ascending=False)
while (df_vif.VIF > 10).any() == True:
    red_df_vif = df_vif.drop(df_vif.index[0])
    df = df[red_df_vif.index]
    df_vif = sklearn_vif(exogs=df.columns, data=df).sort_values(by='VIF', ascending=False)
scaler_biomarkers = scaler_biomarkers[df.columns]
results = pd.DataFrame()
for biomarker in scaler_biomarkers.columns:
    tmp = pd.concat([scaler_biomarkers[biomarker], patient_info], axis=1)
    tmp = tmp.dropna(subset=[biomarker])
    thresh = tmp[biomarker].quantile(q=0.5)
    tmp.loc[tmp[biomarker] > thresh, 'group'] = 'High'
    tmp.loc[tmp[biomarker] <= thresh, 'group'] = 'Low'
    groups = tmp['group']
    ix = (groups == 'High')
    lr = lifelines.statistics.logrank_test(tmp['pfs'][ix], tmp['pfs'][~ix],tmp['status'][ix], tmp['status'][~ix], alpha=.99)
    cph = CoxPHFitter()
    tmp.loc[tmp['group'] == 'High', 'group'] = 1
    tmp.loc[tmp['group'] == 'Low', 'group'] = 0
    cph.fit(tmp[["pfs", "status", 'group']], duration_col="pfs", event_col="status")
    result_i = cph.summary
    result_i.index = [biomarker]
    results = pd.concat([results, result_i])
    results.loc[biomarker, 'km_lr'] = lr.p_value
results['biomarkers'] = results.index
pfs_marker_performance=results
biomarker_005 = list(pfs_marker_performance.loc[pfs_marker_performance['km_lr'] < 0.1, 'biomarkers'].values)
tmp = pd.concat([scaler_biomarkers[biomarker_005], patient_info], axis=1)
tmp = tmp.dropna(subset=biomarker_005)
coxnet_pipe = make_pipeline(StandardScaler(),CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=100))
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)
Xt= tmp[biomarker_005]
y = np.array(tmp[['status','pfs']])
y = list(np.array(tmp[['status', 'pfs']]))
y = np.array([(x[0].astype(bool), x[1]) for x in y], dtype=[('status', bool), ('age', float)])
coxnet_pipe.fit(Xt, y)
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
cv = KFold(n_splits=5, shuffle=True)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    cv=cv,
    error_score=0.5,
    n_jobs=1,
).fit(Xt, y)
cv_results = pd.DataFrame(gcv.cv_results_)
alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)
plt.show()
best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])
non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print(f"Number of non-zero coefficients: {non_zero}")
non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index
_, ax = plt.subplots(figsize=(16, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
plt.tight_layout()
plt.show()
coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=True))
coxnet_pred.set_params(**gcv.best_params_)
coxnet_pred.fit(Xt, y)
coxnet_score = coxnet_pred.score(Xt,y)
fin_model = coxnet_pred.named_steps["coxnetsurvivalanalysis"]
fin_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])
fin_coefs = fin_coefs[fin_coefs['coefficient']!=0]
tmp=pd.concat([scaler_biomarkers[list(fin_coefs.index)], patient_info[['status', 'pfs']]], axis=1)
tmp['comb']=tmp[list(fin_coefs.index)].apply(lambda x:np.dot(x,fin_coefs['coefficient']),axis=1)
biomarker_choose=fin_coefs.index.tolist()
biomarker_weight=fin_coefs['coefficient'].values
cell_df = cell_df.groupby('patient_id').mean()
cell_df['patient_id'] = list(cell_df.index)
cell_df = cell_df.drop_duplicates(subset=['patient_id'], keep='first')
cell_df = cell_df.reset_index(drop=True)
means_biomaker=cell_df.iloc[:, :feature_num].mean()
cell_df.iloc[:, :feature_num] = cell_df.iloc[:, :feature_num].fillna(means_biomaker)
biomarkers = cell_df.iloc[:, :feature_num].copy()
patient_info = cell_df.iloc[:, feature_num:]
scaler = StandardScaler().fit(biomarkers)
scaler_biomarkers = pd.DataFrame(scaler.transform(biomarkers))
scaler_biomarkers.columns = biomarkers.columns
scaler_biomarkers['comb'] = scaler_biomarkers[biomarker_choose].apply(lambda x: np.dot(x, biomarker_weight), axis=1)
sca = pd.DataFrame(scaler.scale_).T
sca.columns = biomarkers.columns
sca=sca[biomarker_choose]
scaler_biomarkers['comb_true'] = biomarkers[biomarker_choose].apply(lambda x: np.dot(x, biomarker_weight/sca.values[0]), axis=1)
results = pd.DataFrame()
for biomarker in ['comb']:
    tmp = pd.concat([scaler_biomarkers[biomarker], patient_info], axis=1)
    tmp = tmp.dropna(subset=[biomarker])
    thresh = find_optimal_cutpoint(tmp, 'pfs', 'status', 'comb')
    tmp.loc[tmp[biomarker] > thresh, 'group'] = 'High'
    tmp.loc[tmp[biomarker] <= thresh, 'group'] = 'Low'
    tmp_stanford=tmp.copy()
    groups = tmp['group']
    ix = (groups == 'High')
    plt.figure(figsize=(10, 8), linewidth=3)
    font_size = 30
    plt.yticks([0, 0.5, 1], fontsize=font_size)
    plt.xlabel('Time (Month)', fontsize=font_size)
    plt.ylabel('PFS', fontsize=font_size)
    plt.ylim(0, 1)
    kmf1 = KaplanMeierFitter()
    kmf1.fit(tmp['pfs'][ix], event_observed=tmp['status'][ix], label='High')
    ax = kmf1.plot_survival_function(show_censors=True, ci_show=False, linewidth=3)
    kmf2 = KaplanMeierFitter()
    kmf2.fit(tmp['pfs'][~ix], event_observed=tmp['status'][~ix], label='Low')
    ax = kmf2.plot_survival_function(ax=ax, show_censors=True, ci_show=False, linewidth=3)
    add_at_risk_counts(kmf1, kmf2, rows_to_show=['At risk'], fontsize=font_size)
    ax.set_xlabel('Time (Month)', fontsize=font_size)
    ax.legend(fontsize=font_size)
    _lg = ax.get_legend()
    _lg.remove()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    lr_ = lifelines.statistics.logrank_test(tmp['pfs'][ix], tmp['pfs'][~ix],tmp['status'][ix], tmp['status'][~ix], alpha=.99)
    cph = CoxPHFitter()
    tmp.loc[tmp['group'] == 'High', 'group'] = 1
    tmp.loc[tmp['group'] == 'Low', 'group'] = 0
    cph.fit(tmp[["pfs", "status", 'group']], duration_col="pfs", event_col="status")
    results = pd.concat([results, cph.summary])
    results.loc['group', 'km_lr'] = lr_.p_value
    hr_data = cph.summary
    formater = "{0:.02f}".format
    a = hr_data[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].copy()
    a[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']] = a[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].applymap(formater)
    a[['exp(coef)']] = a[['exp(coef)']].astype('str')
    a[['exp(coef) lower 95%']] = a[['exp(coef) lower 95%']].astype('str')
    a[['exp(coef) upper 95%']] = a[['exp(coef) upper 95%']].astype('str')
    a['HR'] = 'HR: ' + a['exp(coef)'] + ' (' + a['exp(coef) lower 95%'] + '~' + a['exp(coef) upper 95%'] + ')'
    a['p'] = lr_.p_value
    a['p'] = r"${0:s}$".format(as_si(lr_.p_value, 2))
    a['print'] = a['HR'] + '\n' + 'P-Value: ' + a['p']
    plt.text(20, 0.3, a['print'][0], verticalalignment='center', horizontalalignment='left',fontsize=font_size, color='#000000')
    plt.tight_layout()
    plt.show()
    tmp = tmp[['pfs', 'status', 'comb', 'age', 'sex', 'lot', 'cps', 'grade', 'msi']]
    tmp = tmp.dropna()
    tmp = tmp.reset_index(drop=True)
    tmp.loc[tmp['age'] < 65, 'age'] = 0
    tmp.loc[tmp['age'] >= 65, 'age'] = 1
    tmp.loc[tmp['cps'] > 0, 'cps'] = 1
    cph = CoxPHFitter()
    cph.fit(tmp, 'pfs', event_col='status')
    c = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    formater = "{0:.02f}".format
    c[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']] = c[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].applymap(formater)
    c[['exp(coef)']] = c[['exp(coef)']].astype('str')
    c[['exp(coef) lower 95%']] = c[['exp(coef) lower 95%']].astype('str')
    c[['exp(coef) upper 95%']] = c[['exp(coef) upper 95%']].astype('str')
    c['CI'] = c['exp(coef) lower 95%'] + ',' + c['exp(coef) upper 95%']
    c['HR'] = c['exp(coef)'] + ' (' + c['CI'] + ')'
    c['p_pre'] = c['p'].apply(lambda x: '<0.001' if x < 0.001 else str(round(x, 3)))
    c['p_pre'] = c['p_pre'].apply(lambda x: x + ('0' * (3 - len(x.split('.')[-1]))) if len(x.split('.')[-1]) else x)
    d = c[['HR', 'p_pre']]
    d.columns = ['Hazard Ratio', 'P-value']
    data = d
    data['Hazard'] = data['Hazard Ratio'].apply(lambda x: float(x.split(' ')[0]))
    data['Lower_CI'] = data['Hazard Ratio'].apply(lambda x: float(x.split('(')[1].split(',')[0]))
    data['Upper_CI'] = data['Hazard Ratio'].apply(lambda x: float(x.split(',')[1].split(')')[0]))
    data['Covariate'] = ['Spatial biomarker', 'LOT', 'CPS', 'Sex', 'Age', 'Grade', 'MSI']
    data['P-value-numeric'] = data['P-value'].replace({'<0.001': 0.0005}).astype(float)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(data['Hazard'], range(len(data)), color='blue', s=100)
    ax.errorbar(data['Hazard'], data['Covariate'],
                xerr=[data['Hazard'] - data['Lower_CI'], data['Upper_CI'] - data['Hazard']],
                fmt='o', color='blue', capsize=6, elinewidth=4)
    ax.axvline(x=1, color='gray', linestyle='--')
    ax.set_yticks([])
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()



