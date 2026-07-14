from __future__ import annotations
import io, json, time, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, rankdata, spearmanr

OUT=Path('validation-output'); OUT.mkdir(exist_ok=True)
CODES=['13874A','13874+']; CW={'lev':.45,'asset':.30,'nonrept':.15,'other':.10}; HW={156:.50,52:.30,26:.20}; FW={'1W':5,'4W':20,'8W':40,'13W':65}
S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0 COTValidation/1.0'

def get(url,params=None,n=3):
    err=None
    for i in range(n):
        try:
            r=S.get(url,params=params,timeout=180); r.raise_for_status(); return r
        except Exception as e:
            err=e; time.sleep(2**i)
    raise RuntimeError(f'{url}: {err}')

def cftc_socrata():
    fields=['report_date_as_yyyy_mm_dd','cftc_contract_market_code','market_and_exchange_names','open_interest_all','dealer_positions_long_all','dealer_positions_short_all','asset_mgr_positions_long_all','asset_mgr_positions_short_all','lev_money_positions_long_all','lev_money_positions_short_all','other_rept_positions_long_all','other_rept_positions_short_all','nonrept_positions_long_all','nonrept_positions_short_all']
    p={'$select':','.join(fields),'$where':"cftc_contract_market_code in ('13874A','13874+')",'$order':'report_date_as_yyyy_mm_dd ASC','$limit':5000}
    x=pd.DataFrame(get('https://publicreporting.cftc.gov/resource/gpe5-46if.json',p).json())
    ren={'report_date_as_yyyy_mm_dd':'date','cftc_contract_market_code':'code','market_and_exchange_names':'market','open_interest_all':'oi','dealer_positions_long_all':'dealer_l','dealer_positions_short_all':'dealer_s','asset_mgr_positions_long_all':'asset_l','asset_mgr_positions_short_all':'asset_s','lev_money_positions_long_all':'lev_l','lev_money_positions_short_all':'lev_s','other_rept_positions_long_all':'other_l','other_rept_positions_short_all':'other_s','nonrept_positions_long_all':'nonrept_l','nonrept_positions_short_all':'nonrept_s'}
    x=x.rename(columns=ren); x['date']=pd.to_datetime(x['date'])
    for c in [c for c in ren.values() if c not in ['date','code','market']]: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x,'CFTC Public Reporting Environment/Socrata'

def cftc_zip_fallback():
    urls=['https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip']+[f'https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip' for y in range(2017,datetime.now(timezone.utc).year+1)]
    fs=[]
    for u in urls:
        try:
            z=zipfile.ZipFile(io.BytesIO(get(u,n=2).content)); name=max(z.namelist(),key=lambda n:z.getinfo(n).file_size)
            a=pd.read_csv(z.open(name),low_memory=False); a.columns=a.columns.str.strip()
            d=next(c for c in a if 'Report_Date' in c or 'As_of_Date' in c)
            m={'date':d,'code':'CFTC_Contract_Market_Code','market':'Market_and_Exchange_Names','oi':'Open_Interest_All','dealer_l':'Dealer_Positions_Long_All','dealer_s':'Dealer_Positions_Short_All','asset_l':'Asset_Mgr_Positions_Long_All','asset_s':'Asset_Mgr_Positions_Short_All','lev_l':'Lev_Money_Positions_Long_All','lev_s':'Lev_Money_Positions_Short_All','other_l':'Other_Rept_Positions_Long_All','other_s':'Other_Rept_Positions_Short_All','nonrept_l':'NonRept_Positions_Long_All','nonrept_s':'NonRept_Positions_Short_All'}
            b=a[a[m['code']].astype(str).str.strip().isin(CODES)][list(m.values())].rename(columns={v:k for k,v in m.items()})
            b['date']=pd.to_datetime(b['date'].astype(str).str.zfill(6),format='%y%m%d',errors='coerce') if 'YYMMDD' in d.upper() else pd.to_datetime(b['date'],errors='coerce')
            fs.append(b)
        except Exception as e: print('zip skip',u,e)
    x=pd.concat(fs,ignore_index=True)
    for c in ['oi','dealer_l','dealer_s','asset_l','asset_s','lev_l','lev_s','other_l','other_s','nonrept_l','nonrept_s']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x,'CFTC annual TFF ZIP archives'

def load_cftc():
    x,src=cftc_zip_fallback()
    x=x.dropna(subset=['date','oi']).copy(); x['code']=x['code'].astype(str).str.strip(); x['p']=x['code'].map({'13874A':0,'13874+':1}).fillna(9)
    x=x.sort_values(['date','p']).drop_duplicates('date').drop(columns='p').reset_index(drop=True)
    return x,src

def load_spy():
    try:
        x=pd.read_csv(io.BytesIO(get('https://stooq.com/q/d/l/?s=spy.us&i=d').content)); x.columns=x.columns.str.lower()
        if not {'date','close'}.issubset(x.columns): raise RuntimeError(f'Unexpected Stooq columns: {list(x.columns)}')
        src='Stooq SPY daily close'
    except Exception as e:
        print('Stooq failed',e); p1=int(datetime(2005,1,1,tzinfo=timezone.utc).timestamp()); p2=int((datetime.now(timezone.utc)+timedelta(days=2)).timestamp())
        j=get('https://query1.finance.yahoo.com/v8/finance/chart/SPY',{'period1':p1,'period2':p2,'interval':'1d','events':'history','includeAdjustedClose':'true'}).json()['chart']['result'][0]
        close=j['indicators'].get('adjclose',[{}])[0].get('adjclose') or j['indicators']['quote'][0]['close']; x=pd.DataFrame({'date':pd.to_datetime(j['timestamp'],unit='s',utc=True).tz_convert(None),'close':close}); src='Yahoo Finance SPY adjusted close'
    x['date']=pd.to_datetime(x['date']).dt.normalize(); x['close']=pd.to_numeric(x['close'],errors='coerce'); return x.dropna().sort_values('date').drop_duplicates('date'),src

def rp(s,w):
    return s.rolling(w,min_periods=w).apply(lambda a: rankdata(a,method='average')[-1]/len(a)*100 if not np.isnan(a).any() else np.nan,raw=True)

def rz(s,w):
    def f(a):
        if np.isnan(a).any(): return np.nan
        med=np.median(a); mad=np.median(np.abs(a-med)); return 0 if mad<1e-12 else np.clip((a[-1]-med)/(1.4826*mad),-3,3)
    return s.rolling(w,min_periods=w).apply(f,raw=True)

def score(s,df,p):
    P=0; Z=0
    for w,k in HW.items():
        q=rp(s,w); z=rz(s,w); df[f'{p}_pct{w}']=q; df[f'{p}_rz{w}']=z; P=P+k*q; Z=Z+k*pd.Series(norm.cdf(z)*100,index=s.index)
    df[f'{p}_score']=.7*P+.3*Z; return df[f'{p}_score']

def compute(x):
    d=x.sort_values('date').reset_index(drop=True).copy()
    for n in ['dealer','asset','lev','other','nonrept']: d[f'{n}_net']=(d[f'{n}_l']-d[f'{n}_s'])/d.oi
    sc={n:score(d[f'{n}_net'],d,n) for n in CW}; d['crowd']=sum(CW[n]*sc[n] for n in CW); d['spec_net']=sum(CW[n]*d[f'{n}_net'] for n in CW)
    d['spread_raw']=d.spec_net-d.dealer_net; d['spread_score']=score(d.spread_raw,d,'spread')
    d4=d.spec_net.diff(4); d13=d.spec_net.diff(13); z4=rz(d4,156); z13=rz(d13,156); d['build_score']=.6*norm.cdf(z4)*100+.4*norm.cdf(z13)*100
    d['raw_score']=.7*d.crowd+.2*d.spread_score+.1*d.build_score
    c=pd.DataFrame({n:(sc[n]-50)/50 for n in CW}); num=abs(sum(CW[n]*c[n] for n in CW)); den=sum(CW[n]*c[n].abs() for n in CW); d['agreement']=np.where(den>1e-12,num/den,0); d['final_score']=(50+(d.raw_score-50)*(.6+.4*d.agreement)).clip(0,100)
    return d

def attach(d,p):
    pdts=p.date.to_numpy(dtype='datetime64[ns]'); pc=p.close.to_numpy(float); out=[]
    for _,r in d.iterrows():
        avail=r.date+pd.Timedelta(days=3); i=int(np.searchsorted(pdts,np.datetime64(avail),'left')); q=r.to_dict(); q['available']=avail
        q['price_date']=p.iloc[i].date if i<len(p) else pd.NaT; q['close']=pc[i] if i<len(p) else np.nan
        for lab,n in FW.items(): q[f'fwd_{lab}']=pc[i+n]/pc[i]-1 if i+n<len(p) else np.nan
        out.append(q)
    return pd.DataFrame(out)

def bins(s): return pd.cut(s,[-np.inf,20,40,60,80,np.inf],labels=['0-20','20-40','40-60','60-80','80-100'],right=False,ordered=True)

def summarize(d,period='Full'):
    a=d.dropna(subset=['final_score']).copy(); a['bin']=bins(a.final_score); rows=[]
    for h in FW:
        for b,g in a.groupby('bin',observed=False):
            v=g[f'fwd_{h}'].dropna(); rows.append({'period':period,'horizon':h,'score_bin':str(b),'n':len(v),'mean_return':v.mean(),'median_return':v.median(),'positive_rate':(v>0).mean(),'std_return':v.std()})
    return pd.DataFrame(rows)

def main():
    c,cs=load_cftc(); p,ps=load_spy(); d=attach(compute(c),p); full=summarize(d); allbins=[full]
    valid=d.dropna(subset=['final_score'])
    for name,mask in {'Pre-2020':valid.price_date<pd.Timestamp('2020-01-01'),'2020+':valid.price_date>=pd.Timestamp('2020-01-01'),'Original code only':valid.code=='13874A','Consolidated code':valid.code=='13874+'}.items():
        if mask.sum()>=30: allbins.append(summarize(valid[mask],name))
    br=pd.concat(allbins,ignore_index=True); hs=[]; checks=[]
    for h in FW:
        q=valid[['final_score',f'fwd_{h}']].dropna(); rho,pv=spearmanr(q.final_score,q[f'fwd_{h}']); t=full[full.horizon.eq(h)].set_index('score_bin'); low=t.loc['0-20']; high=t.loc['80-100']; ok=bool(low.mean_return>high.mean_return and rho<0); checks.append(ok)
        hs.append({'horizon':h,'n':len(q),'spearman_rho':rho,'spearman_pvalue':pv,'low_bin_mean':low.mean_return,'high_bin_mean':high.mean_return,'low_minus_high':low.mean_return-high.mean_return,'low_bin_n':int(low.n),'high_bin_n':int(high.n),'directional_check':ok})
    enough=all(x['low_bin_n']>=10 and x['high_bin_n']>=10 for x in hs); passed=sum(checks); verdict='SUPPORTED_AS_SHADOW_FILTER' if passed>=3 and enough else ('PARTIALLY_SUPPORTED_KEEP_SHADOW' if passed>=2 else 'NOT_SUPPORTED_FOR_COMPOSITE')
    summary={'generated_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),'formula_version':'cot-tff-research-informed-v1-frozen','sources':{'cftc':cs,'price':ps},'data':{'cftc_rows':len(d),'valid_score_rows':len(valid),'first_report_date':str(d.date.min().date()),'last_report_date':str(d.date.max().date()),'first_valid_score_date':str(valid.date.min().date()),'code_counts':{str(k):int(v) for k,v in d.code.value_counts().items()},'first_consolidated_date':str(d.loc[d.code.eq('13874+'),'date'].min().date()) if d.code.eq('13874+').any() else None},'validation':{'verdict':verdict,'directional_checks_passed':passed,'directional_checks_total':4,'enough_extreme_observations':enough,'horizons':hs},'guardrails':['No parameter optimization','Tuesday positions first used from Friday/next trading close','SPY price return only','No retroactive Composite or paper-trade changes']}
    keep=['date','available','price_date','code','market','oi','dealer_net','asset_net','lev_net','other_net','nonrept_net','lev_score','asset_score','other_score','nonrept_score','crowd','spread_raw','spread_score','build_score','raw_score','agreement','final_score','close']+[f'fwd_{h}' for h in FW]
    w=d[keep].copy(); w['score_bin']=bins(w.final_score).astype(str); w.to_csv(OUT/'cot_v2_weekly_scores.csv',index=False); br.to_csv(OUT/'cot_v2_bin_results.csv',index=False); pd.DataFrame(hs).to_csv(OUT/'cot_v2_horizon_audit.csv',index=False); (OUT/'cot_v2_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    md=['# COT TFF v2 frozen-formula validation','',f"**Verdict: {verdict}**",'',f"CFTC source: {cs}  ",f"Price source: {ps}  ",f"Valid observations: {len(valid)} ({valid.date.min().date()} to {valid.date.max().date()})",'','|Horizon|N|rho|p|Low 0-20|High 80-100|Difference|Check|','|---|---:|---:|---:|---:|---:|---:|:---:|']
    for x in hs: md.append(f"|{x['horizon']}|{x['n']}|{x['spearman_rho']:.4f}|{x['spearman_pvalue']:.4f}|{x['low_bin_mean']:.2%}|{x['high_bin_mean']:.2%}|{x['low_minus_high']:.2%}|{'PASS' if x['directional_check'] else 'FAIL'}|")
    md+=['','## Guardrails','- Formula was frozen before the run; no weights or thresholds were tuned.','- COT dated Tuesday was aligned to the first SPY close on/after Friday.','- 13874A and later 13874+ consolidated rows are separately identifiable in output.','- This validates a positioning filter, not a standalone trading strategy.']; (OUT/'README.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
