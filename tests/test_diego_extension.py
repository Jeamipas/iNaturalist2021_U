"""Checks of comparability, whole-corpus loading and optimizer continuation."""
import copy
import contextlib
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.diego_muon import DiegoMuon, CoupledMuon
from src.diego_reference.datos import AumentoGPU, LotesMemmap, cargador
from src.diego_reference.entrenamiento import Config, crear_optimizador, programa_lr
from src.diego_reference.modelos import crear_modelo, grupos_ajuste
from src.diego_workflow import plan_from_evidence, state_hash, selected_epoch

ROOT = Path(__file__).resolve().parent.parent


class TestDiegoExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_two_convnext_pairs_have_identical_controls_and_initial_weights(self):
        plan = plan_from_evidence(ROOT)
        self.assertEqual(len(plan),4)
        hashes=[]
        heads=[]
        for left, right in ((plan[0],plan[1]),(plan[2],plan[3])):
            self.assertEqual({k:v for k,v in left.items() if k not in ('id','descripcion','optimizador')},
                             {k:v for k,v in right.items() if k not in ('id','descripcion','optimizador')})
            self.assertEqual(left['modelo'],'convnext_tiny')
            self.assertEqual((left['batch'],left['epocas'],left['paciencia']),(256,16,4))
            torch.manual_seed(left['semilla'])
            a=crear_modelo(left['modelo'],left['preentrenado'],True,0.,num_clases=10000,**left['modelo_kw'])
            torch.manual_seed(right['semilla'])
            b=crear_modelo(right['modelo'],right['preentrenado'],True,0.,num_clases=10000,**right['modelo_kw'])
            self.assertEqual(state_hash(a),state_hash(b))
            self.assertTrue(all(p.requires_grad for p in a.parameters()))
            self.assertEqual(sum(p.numel() for p in a.parameters()),35510128)
            hashes.append(state_hash(a)); heads.append(a.classifier[-1].weight.detach().clone())
            opt=DiegoMuon(a,factor_lr_backbone=left['factor_lr_backbone'])
            matrix_ids={id(p) for p in opt.children[0].param_groups[0]['params']}
            adam_ids={id(p) for g in opt.children[1].param_groups for p in g['params']}
            self.assertFalse(matrix_ids & adam_ids)
            self.assertEqual(matrix_ids | adam_ids,{id(p) for p in a.parameters()})
            self.assertTrue(all(id(p) in adam_ids for p in a.classifier.parameters()))
            self.assertTrue(all(id(p) in adam_ids for n,p in a.named_parameters() if p.ndim<=1 or n.endswith('layer_scale')))
            del a,b,opt
        self.assertFalse(plan[0]['preentrenado']); self.assertTrue(plan[2]['preentrenado'])
        self.assertNotEqual(hashes[0],hashes[1])
        torch.testing.assert_close(heads[0],heads[1],rtol=0,atol=0)

    def test_transfer_lr_and_l2_policy_apply_to_the_same_parameters_in_both_optimizers(self):
        config=plan_from_evidence(ROOT)[2]
        # No pretrained download is needed to inspect parameter-group semantics.
        model=crear_modelo('convnext_tiny',False,True,0.)
        backbone,_=grupos_ajuste(model,'convnext_tiny','completo')
        adam=crear_optimizador(Config(**config),model,backbone)
        muon=crear_optimizador(Config(**{**config,'optimizador':'muon'}),model,backbone)
        adam_groups={id(p):g for g in adam.param_groups for p in g['params']}
        muon_groups={id(p):g for g in muon.param_groups for p in g['params']}
        self.assertEqual(set(adam_groups),set(muon_groups))
        head_ids={id(p) for p in model.classifier.parameters()}
        for name,p in model.named_parameters():
            no_decay=p.ndim<=1 or name.endswith('layer_scale')
            decay=0. if no_decay else config['weight_decay']
            self.assertEqual(adam_groups[id(p)]['weight_decay'],decay)
            self.assertEqual(muon_groups[id(p)]['weight_decay'],decay)
            self.assertAlmostEqual(adam_groups[id(p)]['lr'],.001 if id(p) in head_ids else .0001)
            matrix=name.endswith('weight') and p.ndim>=2 and id(p) not in head_ids
            self.assertAlmostEqual(muon_groups[id(p)]['lr'],.002 if matrix else (.001 if id(p) in head_ids else .0001))

    def test_coupled_l2_enters_muon_gradient_before_orthogonalization(self):
        p = nn.Parameter(torch.tensor([[1.,2.],[3.,4.]]))
        p.grad = torch.full_like(p, .1)
        opt = CoupledMuon([p],lr=0.)
        opt.param_groups[0]['weight_decay'] = .05
        expected = p.grad + p.detach()*.05
        opt.step()
        torch.testing.assert_close(opt.state[p]['momentum'], expected)

    def test_last_short_batch_is_included_and_val_order_is_stable(self):
        class TinyMemmap(LotesMemmap):
            def __init__(self):
                self.idx = np.arange(7)
            def __getitem__(self, indices):
                values = torch.tensor(np.sort(self.idx[np.asarray(indices)]))
                return values, values
        ds = TinyMemmap()
        for training in (True,False):
            first = [y for _,y in cargador(ds,3,training,semilla=42,workers=0)]
            second = [y for _,y in cargador(ds,3,training,semilla=42,workers=0)]
            self.assertEqual([len(y) for y in first], [3,3,1])
            torch.testing.assert_close(torch.cat(first),torch.cat(second))
            self.assertEqual(sorted(torch.cat(first).tolist()),list(range(7)))
            if not training: self.assertEqual(torch.cat(first).tolist(), list(range(7)))

    def test_muon_resume_and_per_step_schedule_match_uninterrupted_updates(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__(); self.body = nn.Linear(4,5); self.fc = nn.Sequential(nn.Dropout(0.),nn.Linear(5,3))
            def forward(self,x): return self.fc(self.body(x).relu())
        torch.manual_seed(42)
        a = Tiny(); b = copy.deepcopy(a)
        x,y = torch.randn(6,4), torch.tensor([0,1,2,0,1,2])
        oa,ob = DiegoMuon(a),DiegoMuon(b)
        sa,sb = torch.optim.lr_scheduler.LambdaLR(oa,programa_lr(8,2)), torch.optim.lr_scheduler.LambdaLR(ob,programa_lr(8,2))
        def step(model,opt,sched):
            opt.zero_grad(set_to_none=True); nn.functional.cross_entropy(model(x),y).backward(); opt.step(); sched.step()
        step(a,oa,sa)
        b.load_state_dict(a.state_dict()); ob.load_state_dict(copy.deepcopy(oa.state_dict())); sb.load_state_dict(sa.state_dict())
        step(a,oa,sa); step(b,ob,sb)
        self.assertEqual(state_hash(a),state_hash(b))
        self.assertEqual(sa.get_last_lr(),sb.get_last_lr())

    def test_checkpoint_epoch_uses_min_delta_not_raw_argmax(self):
        history = [{'epoca':1,'val_top1':.1},{'epoca':2,'val_top1':.1005},{'epoca':3,'val_top1':.099}]
        self.assertEqual(selected_epoch(history,.001),1)

    def test_full_training_loop_resumes_from_epoch_checkpoint_for_both_optimizers(self):
        # Exercise the actual ported loop, checkpoint I/O, early stopping and
        # per-step scheduler on a small CPU fixture. CUDA-specific operations
        # are adapted only in this test; real workers never fall back to CPU.
        import src.diego_reference.entrenamiento as training
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__(); self.conv=nn.Conv2d(3,2,1); self.fc=nn.Sequential(nn.Dropout(0.),nn.Linear(2,10000))
            def forward(self,x): return self.fc(self.conv(x).relu().mean((2,3)))
        class Data(LotesMemmap):
            def __init__(self,split,subconjunto=None):
                self.n_total=6 if split=='train_mini' else 4
                self.y=np.arange(self.n_total)%2
                self.idx=np.arange(self.n_total)
            def __getitem__(self,indices):
                idx=np.sort(self.idx[np.asarray(indices)])
                images=torch.tensor(idx[:,None,None,None]).expand(-1,8,8,3).to(torch.uint8)
                return images,torch.from_numpy(self.y[idx])
        class Augment(nn.Module):
            def __init__(self,*args): super().__init__()
            def forward(self,x,entrenamiento): return x.permute(0,3,1,2).float()/255
        def seed(value):
            random.seed(value); np.random.seed(value); torch.manual_seed(value)
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
                             max_memory_allocated=lambda:0,get_device_name=lambda:'cpu-fixture',
                             get_rng_state=torch.get_rng_state,set_rng_state=lambda value:None)
        proxy=SimpleNamespace(**{k:v for k,v in torch.__dict__.items() if k not in ('device','cuda','autocast')},
                              device=lambda _:torch.device('cpu'),cuda=cuda,
                              autocast=lambda *a,**k:contextlib.nullcontext())
        with tempfile.TemporaryDirectory() as directory:
            for optimizer in ('adam','muon'):
                cfg=Config(id='fixture_'+optimizer,optimizador=optimizer,lr=.001,batch=2,
                           epocas=3,paciencia=10,min_delta=0.)
                common=[patch.object(training,'torch',proxy),patch.object(training,'LotesMemmap',Data),
                        patch.object(training,'AumentoGPU',Augment),patch.object(training,'fijar_semillas',seed),
                        patch.object(training,'crear_modelo',lambda *a,**k:Tiny()),
                        patch.object(training,'grupos_ajuste',lambda m,*a:([m.conv],[])),
                        patch.object(training,'gflops',lambda *a:0.),
                        patch.object(training,'validate_identity',lambda *a:({'fixture':optimizer},None))]
                with contextlib.ExitStack() as stack:
                    for item in common: stack.enter_context(item)
                    for run,limit in (('continuous','1000000000'),('segmented','0')):
                        destination=Path(directory)/optimizer/run
                        with patch.object(training,'RESULTADOS',destination),patch.dict(os.environ,{'NATURALIST_SEGMENT_SECONDS':limit}):
                            if run=='continuous':
                                expected=training.entrenar(cfg,val_sub=4,workers=0)
                            else:
                                for epoch in (1,2):
                                    result=training.entrenar(cfg,val_sub=4,workers=0)
                                    self.assertEqual(result['state'],'NEEDS_RESUME')
                                    self.assertEqual(result['next_epoch'],epoch+1)
                                actual=training.entrenar(cfg,val_sub=4,workers=0)
                        checkpoint=torch.load(destination/cfg.id/'ultimo.pt',weights_only=False)
                        if run=='continuous': expected_state=checkpoint['modelo']
                        else:
                            for name,value in checkpoint['modelo'].items():
                                torch.testing.assert_close(value,expected_state[name],rtol=0,atol=0)
                            for key in ('val_top1','val_top5','val_f1_macro','val_f1_weighted'):
                                self.assertEqual(expected[key],actual[key])

    def test_validation_crop_is_deterministic_and_training_uses_seeded_gpu_policy(self):
        x = torch.randint(0,255,(3,144,144,3),dtype=torch.uint8)
        augment = AumentoGPU('basico')
        torch.testing.assert_close(augment(x,False),augment(x,False),rtol=0,atol=0)
        torch.manual_seed(42); left = augment(x,True)
        torch.manual_seed(42); right = augment(x,True)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        self.assertEqual(left.shape,(3,3,128,128))


if __name__ == '__main__':
    unittest.main()
