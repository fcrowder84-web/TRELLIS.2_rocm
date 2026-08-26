#!/usr/bin/env python3
"""Spirit Legacy TRELLIS.2 production worker."""
from __future__ import annotations
import argparse,gc,json,os,sys,time,traceback
from contextlib import contextmanager
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
os.environ["HSA_ENABLE_DXG_DETECTION"]="1"
os.environ["ATTN_BACKEND"]="sdpa"
os.environ["SPARSE_ATTN_BACKEND"]="sdpa"
os.environ["SPARSE_CONV_BACKEND"]="flex_gemm"
os.environ["PYTORCH_ROCM_ARCH"]="gfx1201"
os.environ["GPU_ARCHS"]="gfx1201"
os.environ["ROCM_HOME"]="/opt/rocm/core-7.14"
os.environ["ROCM_PATH"]="/opt/rocm/core-7.14"
os.environ["HIP_PATH"]="/opt/rocm/core-7.14"
os.environ["TORCH_EXTENSIONS_DIR"]="/home/foster/.cache/torch_extensions_trellis2_gfx1201"
os.environ["PYTHONPATH"]=f"{ROOT}:/home/foster/.cache/torch_extensions_trellis2_gfx1201/uv_rasterize_kernel"
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF",None)

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True),encoding="utf-8"); os.replace(tmp,path)

def update(path,**changes):
    try: state=json.loads(path.read_text(encoding="utf-8"))
    except Exception: state={}
    state.update(changes); state["updated_at"]=time.time(); atomic_json(path,state); return state

def parse_views(values):
    out=[]; seen=set()
    for raw in values:
        if "=" not in raw: raise ValueError(f"Invalid --view {raw!r}; expected LABEL=/path/file.png")
        label,p=raw.split("=",1); label=label.strip().lower().replace(" ","_"); p=Path(p).expanduser().resolve()
        if not label or label in seen: raise ValueError(f"Invalid or duplicate view label: {label}")
        if not p.is_file(): raise FileNotFoundError(f"Missing {label} view: {p}")
        seen.add(label); out.append((label,p))
    if not out: raise ValueError("At least one input image is required")
    return out

@contextmanager
def multi_sampler(sampler,n,steps,mode):
    if n<=1: yield; return
    old=sampler._inference_model; sampler._old_inference_model=old
    try:
        if mode=="stochastic":
            if n>steps: print(f"Warning: {n} images exceeds {steps} sampler steps",flush=True)
            counter={"v":0}
            def wrapped(self,model,x_t,t,cond,**kwargs):
                i=counter["v"]%n; counter["v"]+=1
                return self._old_inference_model(model,x_t,t,cond=cond[i:i+1],**kwargs)
        elif mode=="multidiffusion":
            from trellis2.pipelines.samplers import FlowEulerSampler
            def wrapped(self,model,x_t,t,cond,neg_cond,guidance_strength,guidance_interval,guidance_rescale=0.0,**kwargs):
                preds=[FlowEulerSampler._inference_model(self,model,x_t,t,cond[i:i+1],**kwargs) for i in range(len(cond))]
                pred=sum(preds)/len(preds)
                if guidance_interval[0] <= t <= guidance_interval[1]:
                    neg=FlowEulerSampler._inference_model(self,model,x_t,t,neg_cond,**kwargs)
                    cfg=guidance_strength*pred+(1-guidance_strength)*neg
                    if guidance_rescale>0:
                        xp=self._pred_to_xstart(x_t,t,pred); xc=self._pred_to_xstart(x_t,t,cfg); dims=list(range(1,xp.ndim))
                        sp=xp.std(dim=dims,keepdim=True); sc=xc.std(dim=dims,keepdim=True).clamp_min(1e-6)
                        xr=xc*(sp/sc); cfg=self._xstart_to_pred(x_t,t,guidance_rescale*xr+(1-guidance_rescale)*xc)
                    return cfg
                return pred
        else: raise ValueError(f"Unsupported fusion mode: {mode}")
        sampler._inference_model=wrapped.__get__(sampler,type(sampler)); yield
    finally:
        sampler._inference_model=old
        if hasattr(sampler,"_old_inference_model"): delattr(sampler,"_old_inference_model")

def cond(pipeline,images,res):
    import torch
    c=torch.cat([pipeline.get_cond([im],res)["cond"] for im in images],dim=0)
    return {"cond":c,"neg_cond":torch.zeros_like(c[:1])}

def generate(pipeline,images,seed,resolution,fusion,ss,shape,tex,max_tokens):
    import torch
    ptype={512:"512",1024:"1024_cascade",1536:"1536_cascade"}[resolution]; torch.manual_seed(seed)
    c512=cond(pipeline,images,512); c1024=cond(pipeline,images,1024) if resolution!=512 else None; n=len(images)
    with multi_sampler(pipeline.sparse_structure_sampler,n,ss["steps"],fusion):
        coords=pipeline.sample_sparse_structure(c512,32,1,ss)
    if resolution==512:
        with multi_sampler(pipeline.shape_slat_sampler,n,shape["steps"],fusion):
            slat=pipeline.sample_shape_slat(c512,pipeline.models["shape_slat_flow_model_512"],coords,shape)
        tc=c512; tm=pipeline.models["tex_slat_flow_model_512"]; actual=512
    else:
        with multi_sampler(pipeline.shape_slat_sampler,n,shape["steps"],fusion):
            slat,actual=pipeline.sample_shape_slat_cascade(c512,c1024,pipeline.models["shape_slat_flow_model_512"],pipeline.models["shape_slat_flow_model_1024"],512,resolution,coords,shape,max_num_tokens=max_tokens)
        tc=c1024; tm=pipeline.models["tex_slat_flow_model_1024"]
    with multi_sampler(pipeline.tex_slat_sampler,n,tex["steps"],fusion):
        tslat=pipeline.sample_tex_slat(tc,tm,slat,tex)
    torch.cuda.empty_cache(); return pipeline.decode_latent(slat,tslat,actual)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--job-file",required=True); ap.add_argument("--job-id",required=True); ap.add_argument("--view",action="append",default=[]); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--resolution",type=int,choices=[512,1024,1536],default=512); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--fusion-mode",choices=["stochastic","multidiffusion"],default="multidiffusion")
    ap.add_argument("--ss-steps",type=int,default=12); ap.add_argument("--ss-guidance",type=float,default=7.5); ap.add_argument("--ss-guidance-rescale",type=float,default=.7); ap.add_argument("--ss-rescale-t",type=float,default=5.0)
    ap.add_argument("--shape-steps",type=int,default=12); ap.add_argument("--shape-guidance",type=float,default=7.5); ap.add_argument("--shape-guidance-rescale",type=float,default=.5); ap.add_argument("--shape-rescale-t",type=float,default=3.0)
    ap.add_argument("--tex-steps",type=int,default=12); ap.add_argument("--tex-guidance",type=float,default=1.0); ap.add_argument("--tex-guidance-rescale",type=float,default=0.0); ap.add_argument("--tex-rescale-t",type=float,default=3.0)
    ap.add_argument("--decimation-target",type=int,default=150000); ap.add_argument("--texture-size",type=int,choices=[512,1024,2048,4096],default=1024); ap.add_argument("--max-num-tokens",type=int,default=8192); ap.add_argument("--low-vram",action=argparse.BooleanOptionalAction,default=True); a=ap.parse_args()
    jf=Path(a.job_file).resolve(); od=Path(a.output_dir).resolve(); od.mkdir(parents=True,exist_ok=True); started=time.monotonic(); update(jf,status="running",phase="validating_input",pid=os.getpid(),started_at=time.time())
    try:
        from PIL import Image
        import torch,o_voxel
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        views=parse_views(a.view); images=[]; inputs=[]
        for label,path in views:
            im=Image.open(path)
            if im.mode!="RGBA" or im.getchannel("A").getextrema()[0]==255: raise RuntimeError(f"{label} must be an RGBA PNG with meaningful transparency")
            images.append(im); inputs.append({"label":label,"filename":path.name,"size":[im.width,im.height]})
        names=["sparse_structure_decoder","sparse_structure_flow_model","shape_slat_decoder","shape_slat_flow_model_512","tex_slat_decoder","tex_slat_flow_model_512"]
        if a.resolution>512: names += ["shape_slat_flow_model_1024","tex_slat_flow_model_1024"]
        Trellis2ImageTo3DPipeline.model_names_to_load=names; update(jf,phase="loading_pipeline",view_count=len(images),resolution=a.resolution); t=time.monotonic()
        pipeline=Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B",config_file="pipeline.json"); pipeline.low_vram=a.low_vram; pipeline.cuda(); load=time.monotonic()-t
        ss={"steps":a.ss_steps,"guidance_strength":a.ss_guidance,"guidance_rescale":a.ss_guidance_rescale,"guidance_interval":[.6,1.0],"rescale_t":a.ss_rescale_t}
        shape={"steps":a.shape_steps,"guidance_strength":a.shape_guidance,"guidance_rescale":a.shape_guidance_rescale,"guidance_interval":[.6,1.0],"rescale_t":a.shape_rescale_t}
        tex={"steps":a.tex_steps,"guidance_strength":a.tex_guidance,"guidance_rescale":a.tex_guidance_rescale,"guidance_interval":[.6,.9],"rescale_t":a.tex_rescale_t}
        update(jf,phase="generating",load_seconds=round(load,3)); t=time.monotonic(); meshes=generate(pipeline,images,a.seed,a.resolution,a.fusion_mode,ss,shape,tex,a.max_num_tokens); mesh=meshes[0]; gen=time.monotonic()-t
        update(jf,phase="releasing_models"); pipeline.release_inference_models(); del meshes,pipeline; gc.collect(); torch.cuda.empty_cache()
        update(jf,phase="exporting_glb",generation_seconds=round(gen,3)); t=time.monotonic()
        glb=o_voxel.postprocess.to_glb(vertices=mesh.vertices,faces=mesh.faces,attr_volume=mesh.attrs,coords=mesh.coords,attr_layout=mesh.layout,voxel_size=mesh.voxel_size,aabb=[[-.5,-.5,-.5],[.5,.5,.5]],decimation_target=a.decimation_target,texture_size=a.texture_size,remesh=True,remesh_band=1,remesh_project=0,verbose=True)
        gp=od/"model.glb"; glb.export(str(gp),extension_webp=True); export=time.monotonic()-t
        meta={"job_id":a.job_id,"inputs":inputs,"resolution":a.resolution,"seed":a.seed,"fusion_mode":a.fusion_mode,"view_count":len(images),"samplers":{"sparse":ss,"shape":shape,"texture":tex},"max_num_tokens":a.max_num_tokens,"low_vram":a.low_vram,"decimation_target":a.decimation_target,"texture_size":a.texture_size,"glb":gp.name,"glb_bytes":gp.stat().st_size,"load_seconds":round(load,3),"generation_seconds":round(gen,3),"export_seconds":round(export,3),"total_seconds":round(time.monotonic()-started,3),"completed_at":time.time()}
        mp=od/"metadata.json"; atomic_json(mp,meta); update(jf,status="completed",phase="completed",completed_at=time.time(),glb=str(gp),metadata=str(mp),glb_bytes=meta["glb_bytes"],total_seconds=meta["total_seconds"],exit_code=0); print(f"GLB_PATH={gp}",flush=True); return 0
    except BaseException as e:
        update(jf,status="failed",phase="failed",completed_at=time.time(),error=f"{type(e).__name__}: {e}"[:1000],traceback=traceback.format_exc()[-12000:],exit_code=1); traceback.print_exc(); return 1
if __name__=="__main__": raise SystemExit(main())
