"""Check the Q-value and batch-Huber bounds used in the MsPacman section.

Uses saved data only. No training, gameplay, or written report generation.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def load(name):
    return json.loads((ROOT/name).read_text())


def huber(error):
    return .5*error**2 if error<=1 else error-.5


def main():
    config=load('results/seed1668/config.json')
    with (ROOT/'results/seed1668/training.csv').open() as stream:
        training=list(csv.DictReader(stream))
    trace=load('results/report_checks/diagnostic_trajectory.json')
    batch=config['batch']
    tolerance=1e-6
    rows=[]
    for row in training:
        if int(row['updates'])==0:continue
        mean_error,loss=float(row['td_abs']),float(row['loss'])
        assert math.isfinite(mean_error) and math.isfinite(loss)
        assert mean_error>=0 and loss>=0
        rows.append(dict(episode=int(row['episode']),mean_absolute_error=mean_error,
                         loss=loss,lower=huber(mean_error),upper=huber(batch*mean_error)/batch))
    violations=[r for r in rows if not r['lower']-tolerance<=r['loss']<=r['upper']+tolerance]
    predictions=[q for row in trace for q in row['q_values']]
    assert all(math.isfinite(q) for q in predictions)
    upper=1/(1-config['gamma'])
    training_maxima=[float(row['q_max']) for row in training]
    q_failures=sum(q<0 or q>upper+tolerance for q in predictions)
    q_failures+=sum(q<0 or q>upper+tolerance for q in training_maxima)
    sources=['source/mspacman_dqn.py','results/seed1668/config.json',
             'results/seed1668/training.csv','results/report_checks/diagnostic_trajectory.json']
    result=dict(
        data_scope='Episode-end training logs and one diagnostic game; no new gameplay.',
        huber_bounds=dict(status='PASS' if not violations else 'FAIL',records=len(rows),
                          batch=batch,tolerance=tolerance,violations=violations,
                          formula='h(mean absolute residual) <= mean Huber <= h(batch*mean absolute residual)/batch',
                          example=rows[-1]),
        q_bounds=dict(status='PASS' if not q_failures else 'FAIL',true_lower=0.,true_upper=upper,
                      training_logged_maximum=max(training_maxima),training_games=len(training),
                      diagnostic_predictions=len(predictions),diagnostic_minimum=min(predictions),
                      diagnostic_maximum=max(predictions),violations=q_failures,
                      scope='All-action predictions were saved for one diagnostic game; training logs record maxima, not every individual Q-value.'),
        input_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (ROOT/'results/mathematical_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
    if violations or q_failures:raise SystemExit(1)


if __name__=='__main__':main()
