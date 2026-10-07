"""Short CUDA/BF16 smoke test; synthetic inputs are never project metrics."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from src.diego_reference.datos import AumentoGPU
from src.diego_reference.entrenamiento import Config, crear_optimizador, programa_lr, fijar_semillas
from src.diego_reference.modelos import crear_modelo, grupos_ajuste
from src.diego_workflow import plan_from_evidence, state_hash


def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 is required'
    rows=[]
    for config in plan_from_evidence(ROOT):
        cfg=Config(**config); fijar_semillas(cfg.semilla)
        model=crear_modelo(cfg.modelo,cfg.preentrenado,cfg.bn,cfg.dropout,**cfg.modelo_kw).to('cuda',memory_format=torch.channels_last)
        initial=state_hash(model)
        bb,_=grupos_ajuste(model,cfg.modelo,cfg.ajuste)
        opt=crear_optimizador(cfg,model,bb)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,programa_lr(4,1))
        augment=AumentoGPU('basico').to('cuda')
        torch.cuda.reset_peak_memory_stats()
        # Test the real train batch; transfer/architecture changes can affect VRAM.
        for _ in range(2):
            x=torch.randint(0,255,(cfg.batch,144,144,3),dtype=torch.uint8,device='cuda')
            y=torch.arange(cfg.batch,device='cuda') % 10000
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(augment(x,True))
            loss=torch.nn.functional.cross_entropy(logits.float(),y)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip); opt.step(); sched.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
        model.eval()
        # Validation uses 1024. Catch memory failures before any full-corpus job.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            validation=torch.randint(0,255,(1024,144,144,3),dtype=torch.uint8,device='cuda')
            output=model(augment(validation,False))
            assert output.shape==(1024,10000) and torch.isfinite(output).all()
        rows.append({'id':cfg.id,'initial_hash':initial,'pretrained':cfg.preentrenado,'train_batch':cfg.batch,'val_batch':1024,
                     'synthetic_loss':loss.item(),'gpu':torch.cuda.get_device_name(),'bf16':True,
                     'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30})
        del model,opt,sched,bb,augment,logits,loss,x,y,validation,output; torch.cuda.empty_cache()
    assert rows[0]['initial_hash']==rows[1]['initial_hash'] and rows[2]['initial_hash']==rows[3]['initial_hash']
    assert rows[0]['initial_hash']!=rows[2]['initial_hash']
    print(json.dumps({'purpose':'smoke only; excluded from experiment metrics','checks':rows},indent=2),flush=True)


if __name__=='__main__': main()
