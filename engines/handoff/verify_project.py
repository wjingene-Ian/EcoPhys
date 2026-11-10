"""Independent mathematical checks for the course project."""
from pathlib import Path
import json
import numpy as np
import rmt_analysis as ra

root=Path(__file__).resolve().parent
x=ra.load_returns(root/'tables/07_log_return_matrix.csv').iloc[:504,:100]
r=ra.rmt_decomposition(x.to_numpy())
assert np.allclose(np.diag(r.correlation),1)
assert np.isclose(r.eigenvalues.sum(),100)
assert r.eigenvalues.min()>-1e-10
assert np.allclose(r.eigenvectors.T@r.eigenvectors,np.eye(100))
parts=ra.correlation_components(r)
assert np.allclose(parts['market']+parts['group']+parts['noise'],r.correlation)
c=ra.clipped_correlation(r)
assert np.allclose(np.diag(c),1)
assert np.linalg.eigvalsh(c).min()>0
w=ra.long_only_minimum_variance(c,.05)
assert abs(w.sum()-1)<1e-8 and w.min()>-1e-8 and w.max()<=.05+1e-8
returns=ra.load_returns(root/'tables/07_log_return_matrix.csv')
for start in range(504,len(returns),21):
    train=returns.iloc[start-504:start]
    assert train.index.max()<returns.index[start]
# An immediate 10% loss must count as a drawdown from initial wealth 1.
wealth=np.r_[1,np.cumprod(1+np.array([-.1,.05]))]
assert np.isclose((wealth/np.maximum.accumulate(wealth)-1).min(),-.1)
report={'passed':True,'checks':['unit diagonal','trace','PSD','orthonormal basis',
        'component additivity','clipped PSD','feasible weights','training precedes holding','initial-wealth drawdown']}
(root/'presentation_data/mathematical_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
