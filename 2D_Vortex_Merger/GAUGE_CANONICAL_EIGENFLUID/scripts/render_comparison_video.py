"""Render synchronized 2-row GT/canonical/raw/Laplacian comparison videos."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,Normalize
from verify_analytic import ROOT,CMAP,RED,BLUE,load_gt,write_json,savefig
from audit_saved_derivatives import audit

OUT=ROOT/'outputs/comparison_64_1024'
METHODS=[('gt','GT / DNS'),('canonical','Gauge-canonical Eigenfluid'),('raw','Eigenfluid without gauge fixing'),('laplacian','Analytic Laplacian')]

def load_inputs(ks):
    gt=load_gt(); times=gt['time']; bundle={}; metrics={}; relations={}
    exact=json.loads((OUT/'analytic_eigenrelation.json').read_text())
    for k in ks:
        bundle[k]={}; metrics[k]={}; relations[k]={}
        bundle[k]['gt']=dict(u=gt['velocity_u'],v=gt['velocity_v'],omega=gt['vorticity'])
        for method in ('canonical','raw','laplacian'):
            folder=ROOT/f'outputs/analytic/rollout_k{k}' if method=='laplacian' else OUT/f'k{k}_{method}/evaluation'
            if method!='laplacian':
                audit_file=folder/'derivative_crosscheck.json'
                saved=json.loads(audit_file.read_text()) if audit_file.exists() else {}
                signature=json.loads((folder/'jets.json').read_text())['sha256']
                if saved.get('checkpoint_sha256')!=signature or not saved.get('passed'):
                    audit(folder)
            with np.load(folder/'fields.npz') as z:
                np.testing.assert_allclose(z['time'],times,atol=1e-12,rtol=0)
                bundle[k][method]={key:z[key] for key in ('u','v','omega')}
            data=bundle[k][method]
            mseu=.5*np.mean((data['u']-gt['velocity_u'])**2+(data['v']-gt['velocity_v'])**2,axis=(1,2))
            msew=np.mean((data['omega']-gt['vorticity'])**2,axis=(1,2))
            summary=json.loads((folder/'summary.json').read_text())
            np.testing.assert_allclose([mseu[0],mseu[-1],mseu.mean()],[summary['velocity_mse_initial'],summary['velocity_mse_final'],summary['velocity_mse_mean']],rtol=1e-10)
            metrics[k][method]=dict(velocity_mse=mseu.tolist(),vorticity_mse=msew.tolist(),
                velocity_mse_initial=float(mseu[0]),velocity_mse_final=float(mseu[-1]),velocity_mse_mean=float(mseu.mean()),
                vorticity_mse_initial=float(msew[0]),vorticity_mse_final=float(msew[-1]),vorticity_mse_mean=float(msew.mean()))
            relations[k][method]=exact[str(k)] if method=='laplacian' else json.loads((folder/'eigenrelation.json').read_text())
    return gt,bundle,metrics,relations

def encode(frames_dir,output,fps=5):
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-framerate',str(fps),'-i',str(frames_dir/'frame_%03d.png'),
         '-vf','tpad=start_duration=1:start_mode=clone:stop_duration=1:stop_mode=clone','-r','30','-c:v','libx264','-preset','medium',
         '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(cmd,check=True)
    # Decode the full encoded movie. A zero exit code checks every frame.
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(output),'-f','null','-'],check=True)
    return cmd

def render(ks,gt,bundle,metrics,kind,preview=False):
    dest=OUT/('preview64' if preview else 'video'); dest.mkdir(parents=True,exist_ok=True)
    frames=dest/f'{kind}_frames'; frames.mkdir(exist_ok=True)
    height=8.0 if len(ks)==1 else 13.5
    fig,axes=plt.subplots(len(ks),4,figsize=(24,height),squeeze=False)
    fig.subplots_adjust(left=.055,right=.95,bottom=.085,top=.80 if preview else .865,wspace=.065,hspace=.27)
    fig.patch.set_facecolor('white')
    field_max=max(float(np.max(abs(f['omega']))) for row in bundle.values() for f in row.values())
    speed_max=max(float(np.max(np.hypot(f['u'],f['v']))) for row in bundle.values() for f in row.values())
    norm=TwoSlopeNorm(vmin=-field_max,vcenter=0,vmax=field_max) if kind=='vorticity' else Normalize(vmin=0,vmax=speed_max)
    cmap=CMAP if kind=='vorticity' else 'Blues'
    images={}; labels={}; streams={}
    extent=[0,2*np.pi,0,2*np.pi]
    for row,k in enumerate(ks):
        for col,(method,title) in enumerate(METHODS):
            ax=axes[row,col]; data=bundle[k][method]
            value=data['omega'][0] if kind=='vorticity' else np.hypot(data['u'][0],data['v'][0])
            images[row,col]=ax.imshow(value,origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='bilinear')
            ax.set_title(title,fontsize=16,pad=12)
            ax.set_xlim(0,2*np.pi); ax.set_ylim(0,2*np.pi); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if col==0: ax.set_ylabel(f'{k} modes',fontsize=21,labelpad=20)
            labels[row,col]=ax.text(.5,-.06,'',transform=ax.transAxes,ha='center',va='top',fontsize=12)
    cax=fig.add_axes([.965,.19,.012,.58]); fig.colorbar(images[0,0],cax=cax,label='Vorticity' if kind=='vorticity' else 'Velocity magnitude')
    title=fig.suptitle('',fontsize=21,y=.985)
    fig.text(.5,.024,'Same GT frame 0 projected into each basis; autonomous Navier-Stokes Galerkin rollout thereafter.  No later GT refitting.',ha='center',fontsize=13)
    fig.text(.5,.91 if preview else .94,'Re = 350   |   vortex separation = 0.90   |   128 x 128 evaluation grid   |   identical color scale across all panels',ha='center',fontsize=13)
    for i,t in enumerate(gt['time']):
        for row,k in enumerate(ks):
            for col,(method,_) in enumerate(METHODS):
                data=bundle[k][method]; ax=axes[row,col]
                images[row,col].set_data(data['omega'][i] if kind=='vorticity' else np.hypot(data['u'][i],data['v'][i]))
                if kind=='velocity':
                    if (row,col) in streams:
                        streams[row,col].lines.remove()
                        for patch in list(ax.patches): patch.remove()
                    streams[row,col]=ax.streamplot(gt['x'],gt['y'],data['u'][i],data['v'][i],density=.8,color=RED,linewidth=.55,arrowsize=.6)
                if method=='gt': labels[row,col].set_text('Reference DNS')
                else:
                    m=metrics[k][method]
                    labels[row,col].set_text(f"Velocity MSE {m['velocity_mse'][i]:.3e}  |  Vorticity MSE {m['vorticity_mse'][i]:.3e}")
        title.set_text(f"Vortex merger: {'velocity and streamlines' if kind=='velocity' else 'vorticity'}   |   t = {t:.4f}   |   frame {i+1}/61"+('   |   64-mode preview' if preview else ''))
        fig.savefig(frames/f'frame_{i:03d}.png',dpi=120,facecolor='white')
        if i in (0,60): fig.savefig(dest/f'{kind}_{"initial" if i==0 else "final"}.pdf',facecolor='white')
        if i%10==0: print(f'Rendered {kind}: {i+1}/61',flush=True)
    plt.close(fig)
    name=f'vortex_merger_{kind}_{"64_preview" if preview else "64_1024"}.mp4'
    output=dest/name; cmd=encode(frames,output)
    print(output,flush=True)
    return dict(path=str(output.resolve()),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source_frames=61,source_fps=5,output_fps=30,initial_hold_seconds=1,final_hold_seconds=1,decode_check='passed',ffmpeg_command=cmd)

def report(gt,bundle,metrics,relations,videos):
    rows=[]; relrows=[]; csvrows=[]
    improvements=[]
    for k in (64,1024):
        changes={key:100*(metrics[k]['raw'][key]-metrics[k]['canonical'][key])/metrics[k]['raw'][key]
                 for key in ('velocity_mse_final','vorticity_mse_final','velocity_mse_mean','vorticity_mse_mean')}
        improvements.append(f"- {k} 模态：canonical 相对无规范，末帧速度 MSE 变化 {(-changes['velocity_mse_final']):+.2f}%，末帧涡量 MSE 变化 {(-changes['vorticity_mse_final']):+.2f}%；全时段平均速度 MSE 变化 {(-changes['velocity_mse_mean']):+.2f}%。负号表示误差降低。")
        for method,label in METHODS[1:]:
            m=metrics[k][method]; r=relations[k][method]
            rows.append(f"| {k} | {label} | {m['velocity_mse_initial']:.6e} | {m['velocity_mse_final']:.6e} | {m['velocity_mse_mean']:.6e} | {m['vorticity_mse_final']:.6e} | {m['vorticity_mse_mean']:.6e} |")
            if method=='laplacian':
                ev=r['eigenvalue_relative_error_mean']; rp=r['potential_relation_residual_mean']; ru=r['velocity_relation_residual_mean']
                native_p,native_u=rp,ru
            else:
                ev=r['positive_eigenvalue_relative_error_mean']; rp=r['positive_span']['potential_relation_residual_mean']; ru=r['positive_span']['velocity_relation_residual_mean']
                native_p=r['native']['potential_relation_residual_mean']; native_u=r['native']['velocity_relation_residual_mean']
            relrows.append(f'| {k} | {label} | {ev:.3%} | {rp:.4e} | {ru:.4e} | {native_p:.4e} | {native_u:.4e} |')
            csvrows.append(dict(modes=k,method=method,**{name:value for name,value in m.items() if not isinstance(value,list)},
                positive_eigenvalue_relative_error_mean=ev,positive_psi_residual_mean=rp,positive_u_residual_mean=ru,native_psi_residual_mean=native_p,native_u_residual_mean=native_u))
    with (OUT/'final_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0])); w.writeheader(); w.writerows(csvrows)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,k in enumerate((64,1024)):
        for method,color,label in [('canonical',BLUE,'Canonical'),('raw',RED,'Without gauge fixing'),('laplacian','#3C5488','Laplacian')]:
            axes[row,0].semilogy(gt['time'],metrics[k][method]['velocity_mse'],color=color,label=label)
            axes[row,1].semilogy(gt['time'],metrics[k][method]['vorticity_mse'],color=color,label=label)
        axes[row,0].set(xlabel='Time',ylabel='Velocity MSE',title=f'K={k}'); axes[row,1].set(xlabel='Time',ylabel='Vorticity MSE',title=f'K={k}')
        axes[row,0].legend(); axes[row,1].legend()
    savefig(fig,OUT/'mse_64_1024')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,k in enumerate((64,1024)):
        for method,color,label in [('canonical',BLUE,'Canonical'),('raw',RED,'Without gauge fixing')]:
            r=relations[k][method]; axes[row,0].plot(r['positive_span_eigenvalues'],color=color,label=label)
            axes[row,1].semilogy(np.arange(1,k+1),r['positive_span']['velocity_relation_residual'],color=color,label=label)
        axes[row,0].plot(relations[k]['laplacian']['eigenvalues'],color='#3C5488',linestyle='--',label='Exact Laplacian')
        axes[row,1].axhline(.01,color='#444444',linestyle=':',label='1% residual')
        axes[row,0].set(xlabel='Sorted rank',ylabel='Eigenvalue',title=f'K={k}'); axes[row,1].set(xlabel='Sorted rank',ylabel='Relative velocity eigen-residual',title=f'K={k}')
        axes[row,0].legend(); axes[row,1].legend()
    savefig(fig,OUT/'eigenrelations_64_1024')
    text=f'''# Vortex merger：64 / 1024 模态完整对比

视频为两行四列：上行64模态，下行1024模态；每行依次为GT、canonical Eigenfluid、无规范Eigenfluid、解析Laplacian。各方法仅在第一帧投影GT，随后自主演化到 t=6π。

- [速度与流线视频](video/vortex_merger_velocity_64_1024.mp4)
- [涡量视频](video/vortex_merger_vorticity_64_1024.mp4)
- [最终速度图 PDF](video/velocity_final.pdf)
- [最终涡量图 PDF](video/vorticity_final.pdf)
- [全部数值 CSV](final_metrics.csv)

## MSE：第一帧、最后一帧与全时段

速度 MSE 为全部网格与两个分量的普通均方误差；涡量 MSE 为全部网格的标量均方误差。无归一化。61帧平均值包含首尾帧。

| 模态数 | 方法 | 初始速度 MSE | 最后速度 MSE | 平均速度 MSE | 最后涡量 MSE | 平均涡量 MSE |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

{chr(10).join(improvements)}

![MSE](mse_64_1024.png)

## 是否满足 Laplacian relation

同时检查 `−Δψ=λψ` 和 `−Δu=λu`。定义

`rψ = ||−Δψ−λψ|| / (|λ| ||ψ||)`，`ru = ||−Δu−λu|| / (|λ| ||u||)`。

神经基用解析 Jet2/Jet3 导数求残差；解析 Fourier 基用独立 FFT 微分交叉验证。残差越接近0，越满足特征方程。仅比较特征值大小不足以证明特征函数正确。

另外对全部原始网络模态做了 FFT 与 Jet 导数交叉验证（`derivative_crosscheck.json`）。若近常数势导致单精度采样值的 FFT 微分放大舍入误差，则用相同训练权重重新进行双精度推理复查；该检查不改变训练模型或视频中的轨迹。

“正谱”列是两组去掉势常数后，在各自学到的子空间内做的统一离线 Ritz 诊断；“原生”列使用各自训练时的 gauge 处理。无规范组的近常数模态可使原生 λ 接近0，进而产生很大的相对特征残差。该离线诊断不改变视频中的动力学。

| 模态数 | 方法 | 正特征值平均相对误差 | 正谱 rψ 均值 | 正谱 ru 均值 | 原生 rψ 均值 | 原生 ru 均值 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(relrows)}

**解析 Laplacian 满足特征方程至浮点误差。神经基仍有非零特征残差，不能把全部模态视为已经收敛的精确 Laplacian 特征函数。** 每个模态的 λ、残差和 gauge 比例见各 evaluation 目录下的 `eigenrelation_all_modes.csv`。

![Eigenrelations](eigenrelations_64_1024.png)

## 匹配条件与求解方法

- GT：周期 `[0,2π)^2`，128²，Re350，双涡间距0.90，61帧从0到6π。
- 同一行的有/无规范训练使用相同初值、网络结构、积分点生成规则、优化器和2000轮预算；只切换势均值投影及其伴随。
- 64模态使用宽64、1组周期频率、seed20260907；1024模态复用已有宽32、6组周期频率、seed20260904的 canonical 训练，并从相同初始权重补训无规范组。跨行不仅模态数不同，网络设置也不同；不能将跨行变化纯粹归因于 K。
- 每个正谱神经空间均使用完整速度质量矩阵、curl 刚度矩阵和旋转形式 Galerkin 非线性项。速度 Ritz 变换只改变坐标；没有把尚未收敛的势 λ 直接作为粘性系数。
- 统一使用 float64 ETDRK4，最大时间步0.01；只投影第一帧。解析Laplacian使用独立、去混叠的 Fourier-Galerkin 实现。64模态矩阵自由实现与先前张量实现已核对到约1e-15。
- 视频包含全部61个真实输出时刻，每秒显示5个样本帧，编码为30fps；首尾各停留1秒，未用图像插值虚构流场。
- 色标在所有面板与时间上固定；速度视频的红线是各自速度场的流线。
- 原始输入、GT路径与模型哈希见 [protocol.json](protocol.json)；本次单种子结果不代表跨种子统计显著性。
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    (OUT/'PROGRESS.md').write_text('# Comparison complete\n\nThe full two-row videos and all six evaluations are complete. See [the final report](REPORT.md).\n',encoding='utf-8')
    readme=ROOT/'README.md'
    if readme.exists():
        current=readme.read_text(encoding='utf-8')
        current=current.replace(
            'The [64/1024-mode video comparison](outputs/comparison_64_1024/PROGRESS.md)\nis in progress. Its supervisor trains the missing matched K1024 no-gauge arm,\nthen evaluates and generates the requested two-row videos automatically.',
            'The [completed 64/1024-mode video comparison](outputs/comparison_64_1024/REPORT.md)\ncontains two-row GT/canonical/no-gauge/Laplacian videos, all six final MSE\nresults, and direct potential/velocity Laplacian eigenrelation checks.')
        readme.write_text(current,encoding='utf-8')
    write_json(OUT/'completion.json',dict(complete=True,metrics=metrics,relations=relations,videos=videos,
        layout='2 rows (64,1024) x 4 columns (GT,canonical,raw,Laplacian)',report=str((OUT/'REPORT.md').resolve())))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--only64',action='store_true'); parser.add_argument('--kind',choices=['both','vorticity','velocity'],default='both')
    args=parser.parse_args(); ks=[64] if args.only64 else [64,1024]
    gt,bundle,metrics,relations=load_inputs(ks)
    videos={}
    for kind in ('vorticity','velocity') if args.kind=='both' else (args.kind,):
        videos[kind]=render(ks,gt,bundle,metrics,kind,args.only64)
    if not args.only64:
        if args.kind!='both': raise ValueError('Final completion requires both video kinds')
        report(gt,bundle,metrics,relations,videos)
    else: write_json(OUT/'preview64/completion.json',dict(preview_only=True,videos=videos))

if __name__=='__main__': main()
