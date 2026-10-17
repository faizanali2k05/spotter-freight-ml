from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'reports/metrics.json').read_text())
styles=getSampleStyleSheet();story=[]
def para(t,style='BodyText'):
    story.append(Paragraph(t,styles[style]));story.append(Spacer(1,10))
para('Freight Rate Prediction | Technical Assessment','Title')
para('Methodology and observed holdout results','Heading2')
para(f"Labeled dataset: 48,000 loads. Chronological split: {r['train_rows']:,} training and {r['holdout_rows']:,} holdout rows. Cutoff: {r['cutoff']}. Training ends {r['train_end']}; holdout spans {r['holdout_start']} to {r['holdout_end']}.")
para('Data quality and feature preparation','Heading2')
para('Missing values in labeled development data: '+', '.join(f'{k}={v}' for k,v in r['missing_training'].items())+'. Numeric missingness is handled by CatBoost. Categorical missingness is represented as Unknown. Dates are expanded to calendar and elapsed-time features. Identifiers and the target are excluded from predictors.')
para('Model selection and evaluation','Heading2')
para('Two CatBoost regressors were evaluated using the same chronological holdout. Full model uses market, quote, geographic and route features. December-compatible model uses only the common fields available in the 31-day scenario. Scores below are observed holdout results, not Spotter hidden-set scores.')
rows=[['Model','MAE ($)','RMSE ($)','R2']]
for k,v in r['models'].items():
    m=v['metrics'];rows.append([k,str(m['MAE']),str(m['RMSE']),str(m['R2'])])
t=Table(rows,colWidths=[185,90,90,70]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#064A56')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),8)]));story.append(t);story.append(Spacer(1,15))
para('Final prediction and December scenario','Heading2')
para('After holdout assessment, each model was retrained on all labeled development rows. The full model predicts 12,000 unseen validation loads. The December-compatible model predicts 31 daily rates without fabricating unavailable market index, quote signal or coordinate inputs. The December plot is not evidence of actual December forecasting accuracy.')
chart=ROOT/'scorer_results/candidate_december.png'
if chart.exists():story.append(Image(str(chart),width=6.8*inch,height=3.25*inch))
para('Limitations','Heading2')
para('Concept drift, temporal extrapolation and absent December covariates may affect deployment performance. Spotter calculates final hidden validation metrics after submission. All reported local metrics refer only to the chronological development holdout.')
SimpleDocTemplate(str(ROOT/'reports/assessment_report.pdf'),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=42).build(story)
print('Created reports/assessment_report.pdf')
