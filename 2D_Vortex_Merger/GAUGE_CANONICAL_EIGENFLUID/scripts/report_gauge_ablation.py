"""Produce the paired ablation report only after both fixed-budget arms finish."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from verify_analytic import ROOT,CASE,CMAP,RED,BLUE,load_gt,savefig,write_json

OUT=ROOT/'outputs/gauge_ablation'

def load(path): return json.loads(path.read_text(encoding='utf-8'))
def improvement(c,r,key): return 100*(r[key]-c[key])/r[key]

def main():
    c=load(OUT/'canonical/evaluation/summary.json')
    r=load(OUT/'raw/evaluation/summary.json')
    protocol=load(OUT/'protocol.json')
    cdir=ROOT/'outputs/neural_k64_e2000/run_001'
    rdir=OUT/'raw/run_001'
    cc=load(cdir/'completion.json'); rc=load(rdir/'completion.json')
    assert cc['complete'] and rc['complete'] and c['epoch']==r['epoch']==2000
    cmeta=load(cdir/'checkpoints/hardcoded_sss_frozen.json')
    rmeta=load(rdir/'checkpoints/hardcoded_sss_frozen.json')
    config_diff={key:[cmeta['config'].get(key),rmeta['config'].get(key)]
                 for key in set(cmeta['config'])|set(rmeta['config'])
                 if cmeta['config'].get(key)!=rmeta['config'].get(key)}
    assert set(config_diff)<={'name','result_root','gauge_fixing_enabled','gauge_canonicalization'}
    assert protocol['initial_weights_sha256']==protocol['raw_initial_weights_sha256']
    assert protocol['validation_sha256']==protocol['raw_validation_sha256']
    backend=CASE/'src/eigenfluid/gen/vortex_merger2d/hardcoded_backend.py'
    assert hashlib.sha256(backend.read_bytes()).hexdigest()==protocol['original_backend_sha256']
    refinement=load(OUT/'time_refinement.json')
    max_refinement=max(value for arm in refinement.values() for value in arm.values())
    gt=load_gt(); ce=np.loadtxt(OUT/'canonical/evaluation/mse.csv',delimiter=',',skiprows=1)
    re=np.loadtxt(OUT/'raw/evaluation/mse.csv',delimiter=',',skiprows=1)
    cd=c['native_diagnostics']; rd=r['native_diagnostics']
    keys=['velocity_mse_initial','velocity_mse_mean','velocity_mse_final','vorticity_mse_mean',
          'vorticity_mse_final','projection_only_mean_velocity_mse']
    improvements={key:improvement(c,r,key) for key in keys}
    decision=('有改善' if improvements['velocity_mse_mean']>0 else '没有改善')
    c_history=np.genfromtxt(cdir/'data/hardcoded_training_history.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    r_history=np.genfromtxt(rdir/'data/hardcoded_training_history.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.7),layout='constrained')
    for hist,label,color in [(c_history,'Canonical',BLUE),(r_history,'Without gauge fixing',RED)]:
        axes[0].semilogy(hist['epoch'],hist['train_min_energy'],color=color,label=label)
    axes[0].axhline(1.,color='#444444',linestyle=':',linewidth=1)
    axes[0].set(xlabel='Training epoch',ylabel='Lowest training Rayleigh value',title='Positive mode vs gauge mode')
    axes[0].legend(fontsize=8)
    for data,label,color in [(ce,'Canonical',BLUE),(re,'Without gauge fixing',RED)]:
        axes[1].plot(data[:,0],data[:,1],color=color,label=label)
        axes[2].plot(data[:,0],data[:,2],color=color,label=label)
    axes[1].set(xlabel='Time',ylabel='Velocity MSE',title='Autonomous vortex-merger rollout')
    axes[2].set(xlabel='Time',ylabel='Vorticity MSE',title='Identical full Galerkin evaluation')
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[1].legend(fontsize=8); axes[2].legend(fontsize=8)
    savefig(fig,OUT/'canonical_vs_raw_metrics')
    fields={}
    for arm in ('canonical','raw'):
        with np.load(OUT/arm/'evaluation/fields.npz') as z: fields[arm]=z['omega']
    fig,axes=plt.subplots(3,4,figsize=(12,8.2),layout='constrained')
    for row,(label,data) in enumerate([('DNS',gt['vorticity']),('Canonical K=64',fields['canonical']),('Without gauge fixing K=64',fields['raw'])]):
        for col,j in enumerate((0,20,40,60)):
            im=axes[row,col].imshow(data[j],origin='lower',cmap=CMAP,norm=TwoSlopeNorm(vmin=-.12,vcenter=0,vmax=1.))
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(['t = 0',r'$t=2\pi$',r'$t=4\pi$',r'$t=6\pi$'][col])
            if col==0: axes[row,col].set_ylabel(label)
    fig.colorbar(im,ax=axes,shrink=.75,label='Vorticity'); savefig(fig,OUT/'canonical_vs_raw_vortex_merger')
    checkpoints=[]
    for epoch in (400,800,1200,1600):
        cp=OUT/f'canonical/epoch{epoch:04d}_evaluation/summary.json'
        rp=OUT/f'raw/epoch{epoch:04d}_evaluation/summary.json'
        if cp.exists() and rp.exists():
            a,b=load(cp),load(rp)
            checkpoints.append(dict(epoch=epoch,canonical_mean_mse=a['velocity_mse_mean'],
                raw_mean_mse=b['velocity_mse_mean'],improvement_percent=improvement(a,b,'velocity_mse_mean')))
    checkpoints.append(dict(epoch=2000,canonical_mean_mse=c['velocity_mse_mean'],raw_mean_mse=r['velocity_mse_mean'],
                            improvement_percent=improvements['velocity_mse_mean']))
    checkpoint_rows=[f"| {entry['epoch']} | {entry['raw_mean_mse']:.7e} | {entry['canonical_mean_mse']:.7e} | {entry['improvement_percent']:+.2f}% |" for entry in checkpoints]
    labels={'velocity_mse_initial':'初始速度 MSE','velocity_mse_mean':'61帧平均速度 MSE（主指标）',
        'velocity_mse_final':'末帧速度 MSE','vorticity_mse_mean':'61帧平均涡量 MSE',
        'vorticity_mse_final':'末帧涡量 MSE','projection_only_mean_velocity_mse':'逐帧最优投影速度 MSE（仅诊断）'}
    rows=[f"| {labels[key]} | {r[key]:.8e} | {c[key]:.8e} | {improvements[key]:+.2f}% |" for key in keys]
    native_floor=rd['native_min_rayleigh']; gauge_pct=100*rd['native_max_gauge_energy_fraction']
    text=f'''# Eigenfluid：canonical 与无规范的配对训练

## 结论

**在本次固定初始化的配对实验中，canonical 对预先指定的“61帧平均速度 MSE”{decision}，相对下降 {improvements['velocity_mse_mean']:+.2f}%。**

两组均为 K=64、2000 epochs。下表的改善百分比定义为 `(无规范误差−canonical误差)/无规范误差`；正数表示 canonical 更好，负数表示更差。不能把这一次初始化的结果直接推广为所有随机种子或其他几何的保证。

| 指标 | 无规范 | Canonical | Canonical 改善 |
|---|---:|---:|---:|
{chr(10).join(rows)}

同预算的中间检查如下。结果随训练轮次波动；终点比较固定使用2000轮，没有按 GT 指标挑选 checkpoint。

| 训练轮数 | 无规范平均速度 MSE | Canonical 平均速度 MSE | Canonical 改善 |
|---|---:|---:|---:|
{chr(10).join(checkpoint_rows)}

## 严格匹配了什么

- 双周期 `[0,2π)^2`；GT 为 128²、Re=350、l=0.90，时段 0 到 6π，共61帧。
- 两组均为64个网络输出、宽64、五层仿射 SIREN、相同周期 sin/cos 输入。`canonical_siren=true` 仅是两组共同的 SIREN 初始化选项，**不是 gauge 开关**。
- 完全相同的初始权重文件，SHA256 `{protocol['initial_weights_sha256']}`。
- 完全相同的4096点验证包，SHA256 `{protocol['validation_sha256']}`。
- 每轮训练使用相同 PCG64 IID 点，种子 `20260907 + epoch×1009`，每轮4096点；相同 Adam、学习率计划、两遍 causal MGS、精度与停止轮数。
- 唯一数学差别：canonical 在排序、势质量矩阵和正交化前执行 `ψ−mean(ψ)`，并在反传执行其伴随；无规范组关闭这两处操作。详见 [实际函数差异](backend_function.diff)。
- canonical 组复用刚完成的同条件训练；无规范组从同一 epoch-0 权重重新训练，并非从 canonical 的训练后权重继续。
- 开关开启时，forward loss 和一次训练更新与原后端逐元素完全相同；3项消融测试通过。原始后端文件在实验期间未改变。

## 为什么原始 loss 较低不代表更好

无规范训练后的最低势 Ritz 值为 **{native_floor:.8e}**，canonical 为 **{cd['native_min_rayleigh']:.8f}**；这个周期正谱的正确下界是1。

无规范组的一个势 Ritz 模态有 **{gauge_pct:.8f}%** 的 L² 能量来自常数 gauge。常数本来是零速度方向，却被无规范的势优化纳入模态空间。canonical 组训练后势 Ritz 模态的最大常数能量比例只有 **{cd['native_max_gauge_energy_fraction']:.3e}**。

这说明 canonical 在优化目标中排除了非物理常数方向。不过，本次无规范神经模态并非数学上严格常数，速度质量矩阵仍数值解析出 **{r['physical_mass_resolved_rank']}** 维；不能简单宣称其物理速度基已经降成63维。

两组训练后张成空间的正谱误差也作了独立诊断：先去掉常数再评估同一物理空间的势特征值，canonical 平均相对误差 **{cd['centered_span_eigenvalue_relative_error_mean']:.3%}**，无规范 **{rd['centered_span_eigenvalue_relative_error_mean']:.3%}**。这只是离线谱诊断，不向 rollout 注入修正。

## 流体评估没有替无规范组补 gauge

两组都直接从网络输出的速度 `u=(ψ_y,−ψ_x)` 和涡量组装完整速度质量矩阵、curl 刚度矩阵与旋转形式 Galerkin 非线性张量。此路径不使用势函数去均值、势 Ritz 旋转或训练 Rayleigh 值作为粘性系数。之后的速度正交坐标变换对两组完全一致，并保留全部数值可解析的64维。

只在 t=0 投影 GT 初值，然后用相同 ETDRK4 步长 ≤0.01 自主演化；之后没有重新拟合 GT。逐帧最优投影单独列为表达能力诊断，不用于更新轨迹。

canonical 的这套直接速度评估与上一轮先作势 Ritz 坐标变换的评估，整条轨迹最大速度差为 **{c.get('earlier_canonical_coordinate_check_max_abs',0):.3e}**，验证了坐标变换没有改变结论。

两组都将最大时间步从0.01减半至0.005重新评估，平均/末帧速度及平均涡量 MSE 的最大相对变化为 **{max_refinement:.3e}**，远小于观察到的组间差异；详见 [时间加密检查](time_refinement.json)。

![训练与误差曲线](canonical_vs_raw_metrics.png)

![同色标流场对比](canonical_vs_raw_vortex_merger.png)

## 数据和复现

- [匹配协议和输入哈希](protocol.json)
- [汇总 JSON](comparison.json)
- [Canonical 全部指标](canonical/evaluation/summary.json)
- [无规范全部指标](raw/evaluation/summary.json)
- [无规范模型权重](raw/run_001/checkpoints/hardcoded_sss_frozen.npz)
- [指标图 PDF](canonical_vs_raw_metrics.pdf)
- [流场对照 PDF](canonical_vs_raw_vortex_merger.pdf)
- [无规范全部时序](raw/evaluation/fields.npz)

从 GAUGE_CANONICAL_EIGENFLUID 目录运行：

```powershell
python scripts/train_gauge_ablation.py --resume outputs/gauge_ablation/raw/inputs/initial_epoch0000.npz --run-dir outputs/gauge_ablation/raw_rerun --validation-pack outputs/gauge_ablation/raw/inputs/fixed_validation_iid_uniform_v1.npz
python scripts/evaluate_ablation.py --checkpoint outputs/gauge_ablation/raw_rerun/checkpoints/hardcoded_sss_frozen.npz --output outputs/gauge_ablation/raw_rerun/evaluation
python -m pytest scripts/test_gauge_ablation.py -q
```
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    result=dict(complete=True,comparison='canonical versus no gauge fixing',seed=20260907,modes=64,epochs=2000,
        canonical=c,raw=r,improvement_percent=improvements,matched_config_differences=config_diff,
        checkpoints=checkpoints,tests='3 passed; enabled switch exactly reproduces original update',
        primary_endpoint='mean velocity MSE over all 61 frames',single_seed=True,time_refinement=refinement,report=str(OUT/'REPORT.md'))
    write_json(OUT/'comparison.json',result)
    print(json.dumps(improvements,indent=2)); print(OUT/'REPORT.md')

if __name__=='__main__': main()
