import ni_analysis as A, numpy as np, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.seterr(all='ignore')

rounds,rids=A.load_both('UP',[0,1,2])
mean_h,g2,_=A.herald_g2('UP'); r1=math.asinh(math.sqrt(1/(g2-3)))
oml,phi0=[],{}
for k in rids:
    om,ph,_=A.prefit_omega_phi0(rounds[k][0]['V'],rounds[k][0]['C4'],rounds[k][0]['N']); oml.append(om); phi0[k]=ph
omega=float(np.median(oml))
pooled=A._pool_compensated(rounds,rids,[0,1,2],omega,phi0)
D=16
# per-condition fit
parc={ns:A.fit_pooled_general({ns:pooled[ns]},[ns],r1,D=D) for ns in [0,1,2]}

# per-sector Fisher contribution from the model, peak over phi
phg=np.linspace(0,np.pi,200); dphi=1e-4
frac={}; visb={}
for ns in [0,1,2]:
    p=parc[ns]
    d0=A.output_diag(phg+p['delta'],r1,p['r2'],p['eta_int'],ns,D)
    M=A.apply_detection(d0,p['eta4'],p['nbg'][ns],5,D)
    d1=A.output_diag(phg+p['delta']+dphi,r1,p['r2'],p['eta_int'],ns,D)
    Mp=A.apply_detection(d1,p['eta4'],p['nbg'][ns],5,D)
    dM=(Mp-M)/dphi
    Fn=dM**2/(M+1e-12)                # [phi, sector]
    Ftot=Fn.sum(1)
    ipk=np.argmax(Ftot)               # optimal phase
    fr=Fn[ipk]/Ftot[ipk]
    frac[ns]=fr
    # visibility per sector
    vis=[]
    for n in range(6):
        col=M[:,n]; vis.append((col.max()-col.min())/(col.max()+col.min()) if col.max()>0 else 0)
    visb[ns]=np.array(vis)
    print(f"N1={ns}: Fisher fraction by sector n=0..4: {np.round(fr[:5],3)}  (sum={fr[:5].sum():.2f})")
    print(f"        click sector (n>=1) fraction = {fr[1:].sum():.3f}, resolved-multiphoton (n>=2) = {fr[2:].sum():.3f}")

# figure: Fisher fraction per sector, grouped by N1
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
ns_list=[0,1,2]; nmax=4; x=np.arange(nmax+1); w=0.25
cols=['#4477AA','#EE6677','#228833']
for i,ns in enumerate(ns_list):
    ax1.bar(x+(i-1)*w, frac[ns][:nmax+1], w, color=cols[i], label=f"{'no subtr' if ns==0 else f'N1={ns}'}")
ax1.set_xlabel("photon-number sector n"); ax1.set_ylabel("fraction of Fisher information at optimal phase")
ax1.set_title("(a) Where the phase information lives"); ax1.legend(); ax1.grid(ls=':',alpha=0.4)
ax1.set_xticks(x)
for i,ns in enumerate(ns_list):
    ax2.bar(x[1:]+(i-1)*w, visb[ns][1:nmax+1], w, color=cols[i], label=f"{'no subtr' if ns==0 else f'N1={ns}'}")
ax2.set_xlabel("photon-number sector n"); ax2.set_ylabel("model fringe visibility")
ax2.set_title("(b) Fringe visibility grows with n; subtraction moves it up"); ax2.legend(); ax2.grid(ls=':',alpha=0.4)
ax2.set_xticks(x[1:])
fig.suptitle("Photon subtraction redistributes phase information up the Fock ladder [UP]",fontweight='bold')
fig.tight_layout(); fig.savefig("figures/fig8_sector_information_UP.png",dpi=130); print("saved fig8")
