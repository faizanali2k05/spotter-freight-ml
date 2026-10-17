from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
EXPECTED_ROWS=12000
EXPECTED_IDS={f'TE-{i:06d}' for i in range(1,EXPECTED_ROWS+1)}
DECEMBER_DATES=pd.date_range('2025-12-01','2025-12-31',freq='D')
def fail(msg):raise SystemExit('ERROR: '+msg)
def numeric(frame,col):
    v=pd.to_numeric(frame[col],errors='coerce')
    if v.isna().any() or not np.isfinite(v).all():fail('invalid '+col)
    return v.astype(float)
def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--december-predictions',required=True);p.add_argument('--output-dir',default='scorer_results');a=p.parse_args()
    pred=pd.read_csv(a.predictions)
    if list(pred.columns)!=['load_id','predicted_rate'] or len(pred)!=EXPECTED_ROWS:fail('prediction schema/count')
    if pred.load_id.isna().any() or pred.load_id.duplicated().any() or set(pred.load_id)!=EXPECTED_IDS:fail('prediction IDs')
    if (numeric(pred,'predicted_rate')<=0).any():fail('non-positive predictions')
    dec=pd.read_csv(a.december_predictions)
    if list(dec.columns)!=['pickup','delivery','distance','equipment','weight','date','predicted_rate']:fail('December schema')
    dec['date']=pd.to_datetime(dec.date,errors='coerce')
    if dec.date.isna().any() or dec.date.duplicated().any() or len(dec)!=31 or set(dec.date)!=set(DECEMBER_DATES):fail('December dates')
    for col,v in [('pickup','Lexington'),('delivery','Fort Wayne'),('equipment','Dry Van')]:
        if not dec[col].eq(v).all():fail('December '+col)
    for col,v in [('distance',360),('weight',32000)]:
        if not np.isclose(numeric(dec,col),v).all():fail('December '+col)
    if (numeric(dec,'predicted_rate')<=0).any():fail('December rates')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    dec=dec.sort_values('date')
    fig,ax=plt.subplots(figsize=(10.8,4.8),dpi=180);color='#064A56'
    ax.plot(dec.date,dec.predicted_rate,color=color,linewidth=2.6,marker='o',markersize=3.2)
    floor=float(dec.predicted_rate.min())
    ax.fill_between(dec.date,dec.predicted_rate,floor-max(10.,floor*.02),color=color,alpha=.08)
    ax.set_title('Candidate: December 2025 Predicted Load Rate',loc='left',fontsize=15,fontweight='bold',pad=12)
    ax.set_ylabel('Predicted rate ($)');ax.grid(axis='y',color='#D9E2E4',linewidth=.8)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',rotation=35)
    ax.text(0,-.40,'Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes',transform=ax.transAxes,fontsize=9.5,color='#455A60')
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/'candidate_december.png',bbox_inches='tight');plt.close(fig)
    print('Validated 12,000 final predictions.');print('Validated 31 fixed December predictions.');print('Created chart:',out/'candidate_december.png');print('Final validation metrics are calculated by Spotter after submission.')
if __name__=='__main__':main()
