from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT.parent / 'FIGURES'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def ci95(x):
    x = np.asarray(x, float)
    m = float(np.mean(x))
    if len(x) <= 1:
        return m, m, m
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se

# -------------------------
# Synthetic jurisdiction archetypes
# Values are normalized design inputs, not empirical country estimates.
# -------------------------
ARCHETYPES = {
    'Advanced stable': dict(inflation=.05, fiat_liquidity=.90, digital=.94, regulation=.82,
                            trust=.80, local_capacity=.92, external_dependence=.42,
                            inequality=.34, automation=.62, concentration=.48,
                            breadth=.92, income=.95),
    'Middle-income pressure': dict(inflation=.28, fiat_liquidity=.58, digital=.78, regulation=.68,
                                   trust=.67, local_capacity=.82, external_dependence=.48,
                                   inequality=.46, automation=.40, concentration=.56,
                                   breadth=.82, income=.65),
    'High-inflation skilled': dict(inflation=.72, fiat_liquidity=.28, digital=.73, regulation=.58,
                                   trust=.62, local_capacity=.86, external_dependence=.38,
                                   inequality=.52, automation=.32, concentration=.60,
                                   breadth=.80, income=.48),
    'Low-income low-digital': dict(inflation=.42, fiat_liquidity=.36, digital=.32, regulation=.48,
                                   trust=.55, local_capacity=.56, external_dependence=.55,
                                   inequality=.55, automation=.16, concentration=.60,
                                   breadth=.52, income=.25),
    'Resource-rich concentrated': dict(inflation=.14, fiat_liquidity=.76, digital=.86, regulation=.64,
                                       trust=.64, local_capacity=.90, external_dependence=.32,
                                       inequality=.62, automation=.55, concentration=.82,
                                       breadth=.78, income=.82),
    'Automated transition': dict(inflation=.08, fiat_liquidity=.84, digital=.96, regulation=.78,
                                 trust=.72, local_capacity=.96, external_dependence=.30,
                                 inequality=.44, automation=.82, concentration=.58,
                                 breadth=.94, income=.92),
}

PARAMS = list(next(iter(ARCHETYPES.values())).keys())


def generate_jurisdictions(seed=24680, per_archetype=5):
    rng = np.random.default_rng(seed)
    rows = []
    jid = 0
    for arch, base in ARCHETYPES.items():
        for i in range(per_archetype):
            row = {'jurisdiction_id': jid, 'archetype': arch}
            for k, v in base.items():
                # Small perturbations create heterogeneous jurisdictions while preserving archetype identity.
                sd = .025 if k not in {'inflation'} else .035
                row[k] = float(clip(v + rng.normal(0, sd)))
            rows.append(row)
            jid += 1
    return pd.DataFrame(rows)

JUR = generate_jurisdictions()
JUR.to_csv(OUT / 'jurisdiction_archetypes.csv', index=False)

# -------------------------
# Experiment A: global diffusion under need x capability
# -------------------------
def experiment_a(reps=30, T=80):
    records = []
    n = len(JUR)
    for rep in range(reps):
        rng = np.random.default_rng(10000 + rep)
        # Paired base heterogeneity and period shocks within each replication.
        latent = rng.normal(0, .12, size=n)
        period_shocks = rng.normal(0, .05, size=(T, n))
        adoption = np.full(n, .015) + clip(JUR['digital'].to_numpy() - .5, 0, 1) * .015
        for t in range(T):
            mean_adopt = float(adoption.mean())
            for j, row in JUR.iterrows():
                need = clip(.45*row.inflation + .38*(1-row.fiat_liquidity) + .17*row.inequality)
                capability = clip(.30*row.digital + .25*row.breadth + .20*row.regulation + .15*row.trust + .10*row.local_capacity)
                innovation = clip(.55*row.automation + .45*row.income)
                # Low capability is a hard friction: need alone cannot generate scale.
                score = (-3.15 + 2.65*need + 2.35*capability + .75*innovation +
                         1.25*mean_adopt + latent[j] + period_shocks[t,j])
                target = sigmoid(score) * capability
                adoption[j] = clip(adoption[j] + .085*(target - adoption[j]))
                records.append(dict(rep=rep, period=t, jurisdiction_id=int(row.jurisdiction_id),
                                    archetype=row.archetype, adoption=float(adoption[j]),
                                    need=float(need), capability=float(capability)))
    runs = pd.DataFrame(records)
    runs.to_csv(OUT / 'experiment_A_global_diffusion_runs.csv', index=False)
    final = runs[runs.period == T-1]
    s = final.groupby(['rep','archetype'], as_index=False).adoption.mean()
    summ = s.groupby('archetype').adoption.agg(['mean','std']).reset_index()
    summ['lo'] = summ['mean'] - 1.96*summ['std']/np.sqrt(reps)
    summ['hi'] = summ['mean'] + 1.96*summ['std']/np.sqrt(reps)
    summ.to_csv(OUT / 'summary_A_global_diffusion.csv', index=False)

    # Figure 1
    means = runs.groupby(['period','archetype']).adoption.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8.4,5.3))
    for arch in ARCHETYPES:
        d = means[means.archetype == arch]
        ax.plot(d.period, d.adoption, label=arch, linewidth=1.8)
    ax.set_xlabel('Simulation period')
    ax.set_ylabel('Adoption share')
    ax.set_ylim(0, .7)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Global diffusion: adoption follows both need and capability')
    fig.tight_layout()
    fig.savefig(FIG / 'Fig1_Global_Diffusion.png', dpi=300)
    plt.close(fig)
    return summ

# -------------------------
# Experiment B: exchange islanding
# -------------------------
def islanding_run(seed, mode, T=60, crisis_start=10, crisis_end=42):
    rng = np.random.default_rng(seed)
    # Paired shock arrays are identical across modes for the same seed.
    capacity_shock = rng.normal(0, .035, T)
    demand_shock = rng.normal(0, .025, T)
    trust_shock = rng.normal(0, .02, T)
    data = []
    emergency_readiness = 0.0
    pre_readiness = .72
    for t in range(T):
        crisis = crisis_start <= t <= crisis_end
        ext = .12 if crisis else .88
        local_capacity = clip(.86*(1-.38*(1-ext)) + capacity_shock[t], .08, 1)
        demand = clip(.82 + demand_shock[t], .45, 1)
        fiat_settlement = clip((.70 if not crisis else .095) + .08*ext + rng.normal(0,.015), .02, .95)
        if mode == 'fiat_only':
            unit_readiness = 0.0
        elif mode == 'emergency_htve':
            # Emergency construction has a slow trust/breadth ramp after crisis begins.
            if crisis:
                emergency_readiness = clip(emergency_readiness + .025 + trust_shock[t], 0, .48)
            else:
                emergency_readiness = max(0, emergency_readiness - .01)
            unit_readiness = emergency_readiness
        elif mode == 'prepositioned_htve':
            pre_readiness = clip(pre_readiness + .004*(.76-pre_readiness) + trust_shock[t]*.08, .60, .82)
            unit_readiness = pre_readiness
        else:
            raise ValueError(mode)
        unit_settlement = clip(unit_readiness * (.84*local_capacity + .16*ext), 0, .9)
        # Parallel settlement paths are partially substitutable but bounded by real local capacity.
        settlement = 1 - (1-fiat_settlement)*(1-unit_settlement)
        completion = min(local_capacity, demand, settlement)
        # Bottom access is more sensitive to conventional liquidity; local-unit readiness widens access.
        bottom_access = min(local_capacity, demand,
                            clip(.70*fiat_settlement + .60*unit_settlement - .22*fiat_settlement*unit_settlement, 0, 1))
        data.append(dict(seed=seed, mode=mode, period=t, crisis=int(crisis), external_connectivity=ext,
                         local_capacity=float(local_capacity), fiat_settlement=float(fiat_settlement),
                         unit_readiness=float(unit_readiness), completion=float(completion),
                         bottom_access=float(bottom_access)))
    return data

def experiment_b(reps=100):
    modes = ['fiat_only','emergency_htve','prepositioned_htve']
    rec=[]
    for r in range(reps):
        seed=20000+r
        for m in modes:
            rec.extend(islanding_run(seed,m))
    runs=pd.DataFrame(rec)
    runs.to_csv(OUT/'experiment_B_islanding_runs.csv',index=False)
    cr=runs[runs.crisis==1]
    s=cr.groupby(['seed','mode'])[['completion','bottom_access']].mean().reset_index()
    summ=s.groupby('mode').agg(completion_mean=('completion','mean'),completion_sd=('completion','std'),
                               bottom_access_mean=('bottom_access','mean'),bottom_access_sd=('bottom_access','std')).reset_index()
    for col in ['completion','bottom_access']:
        summ[f'{col}_lo']=summ[f'{col}_mean']-1.96*summ[f'{col}_sd']/np.sqrt(reps)
        summ[f'{col}_hi']=summ[f'{col}_mean']+1.96*summ[f'{col}_sd']/np.sqrt(reps)
    summ.to_csv(OUT/'summary_B_islanding.csv',index=False)
    time=runs.groupby(['period','mode']).completion.mean().reset_index()
    fig,ax=plt.subplots(figsize=(8.4,5.2))
    for m,label in [('fiat_only','Conventional settlement only'),('emergency_htve','HTVE built after disruption'),('prepositioned_htve','Pre-positioned HTVE')]:
        d=time[time['mode']==m]
        ax.plot(d.period,d.completion,label=label,linewidth=2)
    ax.axvspan(10,42,alpha=.12)
    ax.set_xlabel('Simulation period')
    ax.set_ylabel('Completed demand share')
    ax.set_ylim(0, .9)
    ax.set_title('Exchange islanding: pre-positioning matters under external disruption')
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/'Fig2_Exchange_Islanding.png',dpi=300); plt.close(fig)
    return summ

# -------------------------
# Experiment C: functional competition with conventional money
# -------------------------
def experiment_c(reps=100):
    stresses=[.1,.3,.5,.7,.9]
    breadths=[.4,.6,.8,1.0]
    rec=[]
    for r in range(reps):
        rng=np.random.default_rng(30000+r)
        eps=rng.normal(0,.035,size=(len(stresses),len(breadths),3))
        for si,stress in enumerate(stresses):
            for bi,breadth in enumerate(breadths):
                capability=clip(.65*breadth+.35*.74)
                conventional=clip(.92-.78*stress+eps[si,bi,0],.05,.95)
                unit_acceptance=clip(sigmoid(-3.05+4.0*stress+2.35*breadth+eps[si,bi,1]),0,1)*capability
                unit_path=clip(.82*unit_acceptance+.18*breadth,0,1)
                unit_volume=unit_path*(1-conventional)*(.65+.35*breadth)
                fiat_volume=conventional*(1-.18*unit_acceptance)
                total=unit_volume+fiat_volume
                unit_share=unit_volume/total if total>0 else 0
                total_completion=clip(total+.06*breadth+eps[si,bi,2],0,1)
                rec.append(dict(rep=r,stress=stress,breadth=breadth,unit_share=unit_share,
                                unit_acceptance=unit_acceptance,total_completion=total_completion))
    runs=pd.DataFrame(rec); runs.to_csv(OUT/'experiment_C_monetary_competition_runs.csv',index=False)
    summ=runs.groupby(['stress','breadth']).agg(unit_share_mean=('unit_share','mean'),unit_share_sd=('unit_share','std'),
                                                completion_mean=('total_completion','mean')).reset_index()
    summ.to_csv(OUT/'summary_C_monetary_competition.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.2,5.1))
    for b in breadths:
        d=summ[summ.breadth==b]
        ax.plot(d.stress,d.unit_share_mean,marker='o',label=f'Market breadth {int(b*100)}%')
    ax.axhline(.5,linestyle='--',linewidth=1)
    ax.set_xlabel('Conventional monetary stress')
    ax.set_ylabel('HTVE share of completed exchange')
    ax.set_ylim(0,1)
    ax.set_title('HTVE remains complementary unless stress and market breadth are both high')
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'Fig3_Monetary_Competition.png',dpi=300);plt.close(fig)
    return summ

# -------------------------
# Experiment D1: elite capture and governance safeguards
# -------------------------
def experiment_d_capture(reps=100):
    concentrations=[.1,.3,.5,.7,.9]
    safeguards=[.2,.6,.9]
    rec=[]
    for r in range(reps):
        rng=np.random.default_rng(40000+r)
        paired=rng.normal(0,.018,size=(len(concentrations),4))
        for ci,c in enumerate(concentrations):
            for g in safeguards:
                capture=c*(1-g)
                trust=clip(.76-.42*capture+paired[ci,0],0,1)
                market_power=clip(.20+.72*capture+paired[ci,1],0,1)
                social_diversion=clip(.62*capture+paired[ci,2],0,1)
                completion=clip(.66-.115*capture-.055*market_power+.035*trust+paired[ci,3],0,1)
                bottom_access=clip(completion*(.88-.32*capture)+.08*g,0,1)
                rec.append(dict(rep=r,concentration=c,safeguards=g,capture=capture,trust=trust,
                                market_power=market_power,social_diversion=social_diversion,
                                completion=completion,bottom_access=bottom_access))
    runs=pd.DataFrame(rec);runs.to_csv(OUT/'experiment_D1_elite_capture_runs.csv',index=False)
    summ=runs.groupby(['concentration','safeguards']).agg(completion_mean=('completion','mean'),
                    bottom_access_mean=('bottom_access','mean'),trust_mean=('trust','mean'),
                    social_diversion_mean=('social_diversion','mean')).reset_index()
    summ.to_csv(OUT/'summary_D1_elite_capture.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.2,5.1))
    for g in safeguards:
        d=summ[summ.safeguards==g]
        ax.plot(d.concentration,d.bottom_access_mean,marker='o',label=f'Governance safeguards {int(g*100)}%')
    ax.set_xlabel('Concentration of productive/platform control')
    ax.set_ylabel('Bottom-group access')
    ax.set_ylim(0, .85)
    ax.set_title('Elite capture can reverse inclusion unless governance safeguards are strong')
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'Fig4_Elite_Capture.png',dpi=300);plt.close(fig)
    return summ

# D2: levy/social-access mechanism, with operating-viability and real-capacity trade-offs.
def experiment_d_levy(reps=100):
    fees=[0,.0025,.005,.01,.02,.03]
    social_shares=[0,.25,.5,.75,1.0]
    capacities={'spare':.82,'tight':.58}
    rec=[]
    for r in range(reps):
        rng=np.random.default_rng(45000+r)
        # paired seed noise by fee/share cell
        noise=rng.normal(0,.012,size=(len(fees),3))
        for fi,fee in enumerate(fees):
            for si,ss in enumerate(social_shares):
                for cap_name,cap in capacities.items():
                    gross=clip(.66 - 1.45*fee + noise[fi,0],0,1)
                    operating_revenue=gross*fee*(1-ss)
                    social_transfer=gross*fee*ss
                    # Marginal access benefit is concave and capacity-bounded.
                    transfer_effect=(1-np.exp(-42*social_transfer))*(.34+.24*cap)
                    price_pressure=(social_transfer*(1-cap))*5.2
                    bottom_access=clip(.49 + transfer_effect - price_pressure + noise[fi,1],0,cap)
                    completion=min(cap,clip(gross + .08*transfer_effect - .035*fee*100 + noise[fi,2],0,1))
                    rec.append(dict(rep=r,fee=fee,social_share=ss,capacity=cap_name,
                                    completion=completion,bottom_access=bottom_access,
                                    operating_revenue=operating_revenue,social_transfer=social_transfer,
                                    price_pressure=price_pressure))
    runs=pd.DataFrame(rec);runs.to_csv(OUT/'experiment_D2_social_levy_runs.csv',index=False)
    summ=runs.groupby(['fee','social_share','capacity']).agg(completion_mean=('completion','mean'),
                 bottom_access_mean=('bottom_access','mean'),operating_revenue_mean=('operating_revenue','mean'),
                 price_pressure_mean=('price_pressure','mean')).reset_index()
    summ.to_csv(OUT/'summary_D2_social_levy.csv',index=False)

    # Pareto-like visualization for spare capacity: access vs completion, marker size not varied to keep figure simple.
    d=summ[summ.capacity=='spare'].copy()
    fig,ax=plt.subplots(figsize=(8.3,5.2))
    for fee in fees:
        z=d[d.fee==fee]
        ax.plot(z.completion_mean,z.bottom_access_mean,marker='o',label=f'Fee {fee*100:g}%')
    ax.set_xlabel('Completed demand share')
    ax.set_ylabel('Bottom-group access')
    ax.set_title('Social-access levy: redistribution has access, price, and operating trade-offs')
    ax.legend(frameon=False,fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(FIG/'Fig5_Social_Access_Levy.png',dpi=300);plt.close(fig)
    return summ

# -------------------------
# Experiment E: AI/robotics abundance and abundance capture
# -------------------------
def experiment_e(reps=100):
    automations=[0,.25,.5,.75,.95]
    resources=[.25,.5,.75,1.0]
    concentrations=[.1,.5,.9]
    safeguards=[.2,.9]
    rec=[]
    for r in range(reps):
        rng=np.random.default_rng(50000+r)
        noise=rng.normal(0,.012,size=(len(automations),len(resources),3))
        for ai,a in enumerate(automations):
            for ri,res in enumerate(resources):
                tech=a*res
                marginal_cost=float(np.exp(-3.2*tech))
                productive_abundance=clip(.18+.86*tech + noise[ai,ri,0],0,1)
                residual_physical_scarcity=clip(.78-.70*tech + noise[ai,ri,1],.06,1)
                for c in concentrations:
                    for g in safeguards:
                        capture=c*(1-g)
                        # Free access grows with abundance but is withheld by capture.
                        free_access=clip(productive_abundance*(1-.72*capture) - .10*marginal_cost,0,1)
                        settlement_required=clip(residual_physical_scarcity*(.58+.20*(1-g)) +
                                                 (1-free_access)*.34 + .12*capture + noise[ai,ri,2],0,1)
                        access=clip(.50 + .55*free_access - .34*capture - .15*residual_physical_scarcity + .16*g,0,1)
                        # Marginal contribution of exchange/access infrastructure shrinks as universal free access rises,
                        # but can remain material where access is captured or residual scarcity persists.
                        htve_gain=clip(.14*settlement_required*(.55+.45*g)*(1-.55*free_access),0,.15)
                        rec.append(dict(rep=r,automation=a,resource_abundance=res,concentration=c,safeguards=g,
                                        productive_abundance=productive_abundance,marginal_cost=marginal_cost,
                                        residual_scarcity=residual_physical_scarcity,free_access=free_access,
                                        settlement_required=settlement_required,access=access,htve_access_gain=htve_gain))
    runs=pd.DataFrame(rec);runs.to_csv(OUT/'experiment_E_abundance_runs.csv',index=False)
    summ=runs.groupby(['automation','resource_abundance','concentration','safeguards']).agg(
        productive_abundance_mean=('productive_abundance','mean'),free_access_mean=('free_access','mean'),
        settlement_required_mean=('settlement_required','mean'),access_mean=('access','mean'),
        htve_access_gain_mean=('htve_access_gain','mean')).reset_index()
    summ.to_csv(OUT/'summary_E_abundance.csv',index=False)

    # High-resource boundary: settlement need as automation rises.
    d=summ[(summ.resource_abundance==1.0) & (summ.safeguards==.9)]
    fig,ax=plt.subplots(figsize=(8.2,5.1))
    for c in concentrations:
        z=d[d.concentration==c]
        ax.plot(z.automation,z.settlement_required_mean,marker='o',label=f'Control concentration {int(c*100)}%')
    ax.set_xlabel('AI/robotic automation')
    ax.set_ylabel('Share of needs still requiring allocation/settlement')
    ax.set_ylim(0,1)
    ax.set_title('Abundance does not eliminate allocation needs when access remains concentrated')
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'Fig6_Abundance_Capture.png',dpi=300);plt.close(fig)
    return summ


def main():
    a=experiment_a(); b=experiment_b(); c=experiment_c(); d1=experiment_d_capture(); d2=experiment_d_levy(); e=experiment_e()
    # Compact key-results export used by the manuscript generator.
    key=[]
    for _,r in a.iterrows(): key.append({'experiment':'A','condition':r['archetype'],'metric':'final_adoption','value':r['mean']})
    for _,r in b.iterrows():
        key.append({'experiment':'B','condition':r['mode'],'metric':'crisis_completion','value':r['completion_mean']})
        key.append({'experiment':'B','condition':r['mode'],'metric':'crisis_bottom_access','value':r['bottom_access_mean']})
    pd.DataFrame(key).to_csv(OUT/'key_results.csv',index=False)
    print('Experiment A\n',a.to_string(index=False))
    print('\nExperiment B\n',b.to_string(index=False))
    print('\nExperiment C\n',c.to_string(index=False))
    print('\nExperiment D1\n',d1.to_string(index=False))
    print('\nExperiment D2 sample 1% / 50%\n',d2[(d2.fee==.01)&(d2.social_share==.5)].to_string(index=False))
    print('\nExperiment E high abundance boundary\n',e[(e.automation==.95)&(e.resource_abundance==1.0)].to_string(index=False))

if __name__=='__main__':
    main()
