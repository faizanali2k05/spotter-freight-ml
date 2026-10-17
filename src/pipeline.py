from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CATEGORICAL = ['pickup', 'delivery', 'equipment']
COMMON = ['pickup', 'delivery', 'distance', 'equipment', 'weight']
FULL = COMMON + ['pickup_lat','pickup_lon','delivery_lat','delivery_lon','market_index','quote_signal']

def features(df, columns):
    out = df[columns].copy()
    dates = pd.to_datetime(df['date'], errors='raise')
    out['month'] = dates.dt.month
    out['day_of_week'] = dates.dt.dayofweek
    out['day_of_month'] = dates.dt.day
    out['day_of_year'] = dates.dt.dayofyear
    out['days_since_start'] = (dates - pd.Timestamp('2025-01-01')).dt.days
    out['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    for c in CATEGORICAL:
        out[c] = out[c].fillna('Unknown').astype(str)
    for c in out.columns.difference(CATEGORICAL):
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def model(iterations=650):
    return CatBoostRegressor(iterations=iterations, depth=7, learning_rate=0.055,
        loss_function='RMSE', random_seed=42, verbose=False, thread_count=4,
        l2_leaf_reg=5)

def metrics(y, p):
    return {'MAE':round(float(mean_absolute_error(y,p)),3),
            'RMSE':round(float(np.sqrt(mean_squared_error(y,p))),3),
            'R2':round(float(r2_score(y,p)),5)}

def main():
    train = pd.read_csv(DATA/'train_test.csv').sort_values('date', kind='stable').reset_index(drop=True)
    val = pd.read_csv(DATA/'validation.csv')
    dec = pd.read_csv(DATA/'december_chart_inputs.csv')
    template = pd.read_csv(DATA/'validation_predictions_template.csv')
    assert len(val)==len(template)==12000
    assert train['posted_rate'].notna().all()
    assert set(val.load_id)==set(template.load_id)
    # Strict chronological holdout: last 20% of development rows, with dates not shared across folds.
    unique_dates = np.sort(pd.to_datetime(train.date).unique())
    cutoff = unique_dates[int(len(unique_dates)*.8)]
    train_mask = pd.to_datetime(train.date) < cutoff
    valid_mask = ~train_mask
    y = train.posted_rate
    results = {'cutoff':str(pd.Timestamp(cutoff).date()),'train_rows':int(train_mask.sum()),'holdout_rows':int(valid_mask.sum()),
               'train_end':str(train.loc[train_mask,'date'].max()),'holdout_start':str(train.loc[valid_mask,'date'].min()),
               'holdout_end':str(train.loc[valid_mask,'date'].max()),'models':{},
               'missing_training':{k:int(v) for k,v in train.isna().sum().items() if v}}
    for name, cols in [('full',FULL),('december_compatible',COMMON)]:
        x = features(train, cols)
        m = model()
        m.fit(x.loc[train_mask],y.loc[train_mask],cat_features=CATEGORICAL,
              eval_set=(x.loc[valid_mask],y.loc[valid_mask]),early_stopping_rounds=75)
        pred = np.maximum(m.predict(x.loc[valid_mask]),1)
        results['models'][name] = {'metrics':metrics(y.loc[valid_mask],pred),'best_iteration':int(m.best_iteration_)}
        # Retrain on all labeled development rows using selected iteration count.
        final = model(max(100,int(m.best_iteration_)+1))
        final.fit(x,y,cat_features=CATEGORICAL)
        (ROOT/'models').mkdir(exist_ok=True)
        final.save_model(str(ROOT/'models'/f'{name}.cbm'))
        if name=='full':
            p = np.maximum(final.predict(features(val,cols)),1)
            prediction_map = dict(zip(val.load_id,p))
            template['predicted_rate'] = template.load_id.map(prediction_map)
            template.to_csv(ROOT/'validation_predictions.csv',index=False,float_format='%.4f')
        else:
            dec['predicted_rate'] = np.maximum(final.predict(features(dec,cols)),1)
            dec.to_csv(DATA/'december_chart_inputs.csv',index=False,float_format='%.4f')
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/'metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
    print('Created validation_predictions.csv and completed December CSV.')

if __name__=='__main__':main()
