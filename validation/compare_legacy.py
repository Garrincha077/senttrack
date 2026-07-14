from __future__ import annotations
import io, json, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import validate_cot_v2 as v

OUT=Path('validation-output'); OUT.mkdir(exist_ok=True)
LH={26:.50,52:.30,156:.20}

def load_legacy():
    urls=['https://www.cftc.gov/files/dea/history/deacot1986_2016.zip']+[f'https://www.cftc.gov/files/dea/history/deacot{y}.zip' for y in range(2017,datetime.now(timezone.utc).year+1)]
    fs=[]; notes=[]
    for u in urls:
        try:
            z=zipfile.ZipFile(io.BytesIO(v.get(u,n=2).content)); name=max(z.namelist(),key=lambda n:z.getinfo(n).file_size)
            a=pd.read_csv(z.open(name),low_memory=False); a.columns=a.columns.str.strip(); d=next(c for c in a if 'Report_Date' in c or 'As_of_Date' in c)
            m={'date':d,'code':'CFTC_Contract_Market_Code','market':'Market_and_Exchange_Names','oi':'Open_Interest_All','large_l':'Noncommercial_Positions_Long_All','large_s':'Noncommercial_Positions_Short_All','comm_l':'Commercial_Positions_Long_All','comm_s':'Commercial_Positions_Short_All','small_l':'Nonreportable_Positions_Long_All','small_s':'Nonreportable_Positions_Short_All'}
            b=a[a[m['code']].astype(str).str.strip().eq('13874A')][list(m.values())].rename(columns={vv:k for k,vv in m.items()})
            b['date']=pd.to_datetime(b.date.astype(str).str.zfill(6),format='%y%m%d',errors='coerce') if 'YYMMDD' in d.upper() else pd.to_datetime(b.date,errors='coerce')
            fs.append(b); notes.append(f'{u}: {len(b)} rows')
        except Exception as e: notes.append(f'{u}: skipped {type(e).__name__}: {e}')
    x=pd.concat(fs,ignore_index=True).dropna(subset=['date','oi'])
    for c in ['oi','large_l','large_s','comm_l','comm_s','small_l','small_s']: x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    return x,notes

def pct_score(s):
    out=0
    for w,k in LH.items(): out=out+k*v.rp(s,w)
    return out

def compute_legacy(x):
    d=x.copy()
    d['large_net']=(d.large_l-d.large_s)/d.oi; d['comm_net']=(d.comm_l-d.comm_s)/d.oi; d['small_net']=(d.small_l-d.small_s)/d.oi
    d['large_score']=pct_score(d.large_net); d['comm_raw_score']=pct_score(d.comm_net); d['comm_directed_score']=100-d.comm_raw_score; d['small_score']=pct_score(d.small_net)
    d['legacy_score']=.42*d.large_score+.40*d.comm_directed_score+.18*d.small_score
    return d

def bin_summary(d,score_col,period='Full'):
    a=d.dropna(subset=[score_col]).copy(); a['score_bin']=v.bins(a[score_col]); rows=[]
    for h in v.FW:
        for b,g in a.groupby('score_bin',observed=False):
            z=g[f'fwd_{h}'].dropna(); rows.append({'model':'Legacy COT v1','period':period,'horizon':h,'score_bin':str(b),'n':len(z),'mean_return':z.mean(),'median_return':z.median(),'positive_rate':(z>0).mean(),'std_return':z.std()})
    return pd.DataFrame(rows)

def horizon_audit(d,score_col):
    full=bin_summary(d,score_col); rows=[]
    for h in v.FW:
        q=d[[score_col,f'fwd_{h}']].dropna(); rho,pv=spearmanr(q[score_col],q[f'fwd_{h}']); t=full[full.horizon.eq(h)].set_index('score_bin'); lo=t.loc['0-20']; hi=t.loc['80-100']; ok=bool(lo.mean_return>hi.mean_return and rho<0)
        rows.append({'model':'Legacy COT v1','horizon':h,'n':len(q),'spearman_rho':rho,'spearman_pvalue':pv,'low_bin_mean':lo.mean_return,'high_bin_mean':hi.mean_return,'low_minus_high':lo.mean_return-hi.mean_return,'low_bin_n':int(lo.n),'high_bin_n':int(hi.n),'directional_check':ok})
    return pd.DataFrame(rows),full

def tff_components():
    d=pd.read_csv(OUT/'cot_v2_weekly_scores.csv',parse_dates=['date','price_date'])
    comps=['lev_score','asset_score','other_score','nonrept_score','crowd','spread_score','build_score','raw_score','agreement','final_score']; rows=[]
    for c in comps:
        for h in v.FW:
            q=d[[c,f'fwd_{h}']].dropna(); rho,pv=spearmanr(q[c],q[f'fwd_{h}']); rows.append({'component':c,'horizon':h,'n':len(q),'spearman_rho':rho,'pvalue':pv})
    return pd.DataFrame(rows)

def main():
    x,notes=load_legacy(); p,ps=v.load_spy(); d=v.attach(compute_legacy(x),p); audit,bins=horizon_audit(d,'legacy_score')
    pre=d[d.price_date<pd.Timestamp('2020-01-01')]; post=d[d.price_date>=pd.Timestamp('2020-01-01')]; allbins=[bins]
    if pre.legacy_score.notna().sum()>=30: allbins.append(bin_summary(pre,'legacy_score','Pre-2020'))
    if post.legacy_score.notna().sum()>=30: allbins.append(bin_summary(post,'legacy_score','2020+'))
    bins=pd.concat(allbins,ignore_index=True); comp=tff_components(); passed=int(audit.directional_check.sum()); enough=bool(((audit.low_bin_n>=10)&(audit.high_bin_n>=10)).all()); verdict='SUPPORTED_FOR_COMPOSITE' if passed>=3 and enough else ('PARTIALLY_SUPPORTED_KEEP_ACTIVE_WITH_CAUTION' if passed>=2 else 'NOT_VALIDATED_AS_CONTRARIAN_COMPONENT')
    tff=json.loads((OUT/'cot_v2_summary.json').read_text())
    summary={'generated_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),'legacy':{'verdict':verdict,'rows':len(d),'valid_score_rows':int(d.legacy_score.notna().sum()),'first_report_date':str(d.date.min().date()),'last_report_date':str(d.date.max().date()),'price_source':ps,'archive_notes':notes,'horizons':audit.to_dict(orient='records')},'tff':tff,'decision_rule':'Neither model is migrated or reweighted by this diagnostic run.'}
    d[['date','available','price_date','large_net','comm_net','small_net','large_score','comm_directed_score','small_score','legacy_score','close']+[f'fwd_{h}' for h in v.FW]].to_csv(OUT/'legacy_v1_weekly_scores.csv',index=False)
    bins.to_csv(OUT/'legacy_v1_bin_results.csv',index=False); audit.to_csv(OUT/'legacy_v1_horizon_audit.csv',index=False); comp.to_csv(OUT/'tff_component_correlations.csv',index=False); (OUT/'cot_comparison_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps({'legacy_verdict':verdict,'legacy_horizons':audit.to_dict(orient='records'),'tff_verdict':tff['validation']['verdict']},indent=2,default=str))
if __name__=='__main__': main()
