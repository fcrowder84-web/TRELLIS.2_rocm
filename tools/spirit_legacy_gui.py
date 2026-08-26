#!/usr/bin/env python3
from __future__ import annotations
import json,os,random,re,shutil,subprocess,time
from datetime import datetime
from pathlib import Path
import gradio as gr
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path(os.environ.get('TRELLIS_PYTHON','/home/foster/trellis2-env/bin/python'))
WORKER=Path(os.environ.get('TRELLIS_STUDIO_WORKER',str(ROOT/'tools'/'spirit_legacy_worker.py')))
OUT=Path(os.environ.get('TRELLIS_STUDIO_OUTPUTS','/home/foster/trellis2-outputs/studio')); OUT.mkdir(parents=True,exist_ok=True)
MAX_SEED=2147483647
VIEWS=[('Front','front'),('Back','back'),('Left','left'),('Right','right'),('Top','top'),('Bottom','bottom'),('Extra View 1','extra_1'),('Extra View 2','extra_2')]
PRESETS={
 'Phase 8 Safe':('512','multidiffusion',12,12,12,150000,'1024',8192),
 'Fast Preview':('512','stochastic',8,8,8,100000,'1024',8192),
 'Game Asset':('512','multidiffusion',12,12,12,250000,'2048',8192),
 'High-Res Master (Experimental)':('1024','multidiffusion',16,16,16,750000,'4096',24576),
 'Ultra 1536 (Experimental)':('1536','multidiffusion',20,20,20,1000000,'4096',32768),
 'Custom':None}

def clean(v):
 v=re.sub(r'[^A-Za-z0-9._-]+','_', (v or 'asset').strip()).strip('._-'); return v[:80] or 'asset'
def preset(name):
 p=PRESETS.get(name); editable=p is None
 if editable: return tuple(gr.update(interactive=True) for _ in range(8))
 return tuple(gr.update(value=v,interactive=False) for v in p)
def state(path):
 try:return json.loads(path.read_text(encoding='utf-8'))
 except Exception:return {}
def verify(label,path):
 im=Image.open(path)
 if im.mode!='RGBA' or im.getchannel('A').getextrema()[0]==255:
  raise gr.Error(f'{label}: use an RGBA PNG with meaningful transparency.')

def run(asset,category,preset_name,resolution,fusion,randomize,seed,ss_steps,ss_g,ss_gr,ss_t,shape_steps,shape_g,shape_gr,shape_t,tex_steps,tex_g,tex_gr,tex_t,decimation,texture,tokens,*files):
 chosen=[]
 for (display,label),path in zip(VIEWS,files):
  if path: verify(display,path); chosen.append((display,label,Path(path)))
 if not chosen: raise gr.Error('Select at least one reference image.')
 seed=random.randint(0,MAX_SEED) if randomize else int(seed)
 stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); job_id=f'{clean(asset)}_{stamp}_{seed}'; jobdir=OUT/clean(category or 'Other')/job_id; inp=jobdir/'inputs'; inp.mkdir(parents=True)
 viewargs=[]; inputs=[]
 for _,label,src in chosen:
  dst=inp/f'{label}{src.suffix.lower() or ".png"}'; shutil.copy2(src,dst); viewargs += ['--view',f'{label}={dst}']; inputs.append({'label':label,'original_filename':src.name,'stored_filename':dst.name})
 (jobdir/'request.json').write_text(json.dumps({'asset_name':asset,'category':category,'preset':preset_name,'resolution':int(resolution),'fusion_mode':fusion,'seed':seed,'inputs':inputs,'created_at':time.time()},indent=2),encoding='utf-8')
 jf=jobdir/'job.json'; logp=jobdir/'worker.log'
 cmd=[str(PYTHON),str(WORKER),'--job-file',str(jf),'--job-id',job_id,*viewargs,'--output-dir',str(jobdir),'--resolution',str(resolution),'--seed',str(seed),'--fusion-mode',fusion,'--ss-steps',str(int(ss_steps)),'--ss-guidance',str(ss_g),'--ss-guidance-rescale',str(ss_gr),'--ss-rescale-t',str(ss_t),'--shape-steps',str(int(shape_steps)),'--shape-guidance',str(shape_g),'--shape-guidance-rescale',str(shape_gr),'--shape-rescale-t',str(shape_t),'--tex-steps',str(int(tex_steps)),'--tex-guidance',str(tex_g),'--tex-guidance-rescale',str(tex_gr),'--tex-rescale-t',str(tex_t),'--decimation-target',str(int(decimation)),'--texture-size',str(texture),'--max-num-tokens',str(int(tokens)),'--low-vram']
 with logp.open('w',encoding='utf-8',buffering=1) as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,text=True)
  while proc.poll() is None:
   s=state(jf); yield f"Job: {job_id}\nViews: {len(chosen)} ({', '.join(x[1] for x in chosen)})\nSeed: {seed}\nResolution: {resolution}\nFusion: {fusion}\nPhase: {s.get('phase','starting')}",None,None; time.sleep(1)
 s=state(jf)
 if proc.returncode or s.get('status')!='completed':
  try: tail='\n'.join(logp.read_text(encoding='utf-8').splitlines()[-25:])
  except Exception: tail=''
  yield f"FAILED: {s.get('error',f'worker exit {proc.returncode}')}\n\n{tail}",None,str(jf); return
 gp=Path(s['glb']); mp=Path(s.get('metadata',jobdir/'metadata.json')); yield f"Completed: {job_id}\nViews: {len(chosen)}\nSeed: {seed}\nResolution: {resolution}\nFusion: {fusion}\nTotal seconds: {s.get('total_seconds')}\nOutput: {gp}",str(gp),str(mp)

with gr.Blocks(title='Spirit Legacy TRELLIS Studio') as demo:
 gr.Markdown('# Spirit Legacy TRELLIS Studio\nSelect any views you have. Filenames do not matter. Transparent RGBA PNGs are required on this ROCm setup.')
 with gr.Row():
  asset=gr.Textbox(label='Asset Name',placeholder='Greyhaven_Blacksmith_01',scale=2); category=gr.Dropdown(['Building','Tree','Prop','Weapon','Armor','Character','Creature','Environment','Other'],value='Prop',label='Category',allow_custom_value=True); preset_name=gr.Dropdown(list(PRESETS),value='Phase 8 Safe',label='Quality Preset')
 gr.Markdown('### Reference views')
 files=[]
 for i in range(0,8,4):
  with gr.Row():
   for display,_ in VIEWS[i:i+4]: files.append(gr.File(label=display,file_types=['.png'],type='filepath'))
 with gr.Row():
  resolution=gr.Radio(['512','1024','1536'],value='512',label='Generation Resolution',interactive=False); fusion=gr.Radio(['multidiffusion','stochastic'],value='multidiffusion',label='Multi-View Fusion',interactive=False); randomize=gr.Checkbox(True,label='Randomize Seed'); seed=gr.Slider(0,MAX_SEED,42,step=1,label='Seed')
 with gr.Accordion('Advanced Settings',open=False):
  gr.Markdown('Choose **Custom** to edit preset-managed values. Guidance controls remain adjustable for fine tuning.')
  with gr.Row(): ss_steps=gr.Slider(4,40,12,step=1,label='Sparse Steps',interactive=False); ss_g=gr.Slider(0,20,7.5,step=.1,label='Sparse Guidance'); ss_gr=gr.Slider(0,1,.7,step=.05,label='Sparse Guidance Rescale'); ss_t=gr.Slider(0,10,5,step=.1,label='Sparse Rescale T')
  with gr.Row(): shape_steps=gr.Slider(4,40,12,step=1,label='Shape Steps',interactive=False); shape_g=gr.Slider(0,20,7.5,step=.1,label='Shape Guidance'); shape_gr=gr.Slider(0,1,.5,step=.05,label='Shape Guidance Rescale'); shape_t=gr.Slider(0,10,3,step=.1,label='Shape Rescale T')
  with gr.Row(): tex_steps=gr.Slider(4,40,12,step=1,label='Texture Steps',interactive=False); tex_g=gr.Slider(0,10,1,step=.1,label='Texture Guidance'); tex_gr=gr.Slider(0,1,0,step=.05,label='Texture Guidance Rescale'); tex_t=gr.Slider(0,10,3,step=.1,label='Texture Rescale T')
  with gr.Row(): decimation=gr.Slider(100000,1500000,150000,step=10000,label='GLB Decimation',interactive=False); texture=gr.Radio(['512','1024','2048','4096'],value='1024',label='Texture Size',interactive=False); tokens=gr.Slider(4096,49152,8192,step=1024,label='Max Tokens',interactive=False)
 go=gr.Button('Generate 3D Asset',variant='primary'); status=gr.Textbox(label='Job Status',lines=8,interactive=False)
 with gr.Row(): glb=gr.File(label='Generated GLB'); meta=gr.File(label='Generation Metadata')
 preset_name.change(preset,[preset_name],[resolution,fusion,ss_steps,shape_steps,tex_steps,decimation,texture,tokens])
 go.click(run,[asset,category,preset_name,resolution,fusion,randomize,seed,ss_steps,ss_g,ss_gr,ss_t,shape_steps,shape_g,shape_gr,shape_t,tex_steps,tex_g,tex_gr,tex_t,decimation,texture,tokens,*files],[status,glb,meta])
if __name__=='__main__': demo.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0',server_port=int(os.environ.get('TRELLIS_STUDIO_PORT','7860')),show_error=True,allowed_paths=[str(OUT)])
