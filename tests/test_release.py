import ast
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from recap.context import ReCAPContext
from recap.assets import sha256, verify, safe_path
from recap.cli import build_command

ROOT=Path(__file__).resolve().parents[1]
class ContextTests(unittest.TestCase):
    def test_budget_eviction_and_immutability(self):
        ctx=[9]*1280; anchor=[8]*93;s=ReCAPContext(ctx,anchor)
        ctx[0]=0;anchor[0]=0
        for t in range(20):
            self.assertEqual(s.append([t]*80,[100+t]*13),t>=6)
            p=s.prompt();self.assertLessEqual(len(p),1931)
            self.assertEqual(p[:1280],[9]*1280);self.assertEqual(p[1280:1373],[8]*93)
        self.assertEqual(s.prompt()[1373:1453],[14]*80)
    def test_complete_blocks(self):
        with self.assertRaises(ValueError):ReCAPContext([],[])
        s=ReCAPContext([1]*1280,[2]*93)
        with self.assertRaises(ValueError):s.append([3]*79,[4]*13)
        with self.assertRaises(ValueError):ReCAPContext([1]*1280,[2]*93,-1)
    def test_matches_actual_paper_schedule(self):
        # Execute the preserved function using a deterministic fake sampler; compare
        # every prompt, including the first eviction and the full 64-frame horizon.
        tree=ast.parse((ROOT/'src/recap/reference/infer_compare.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_recap')
        for window in [1,6,12]:
            prompts=[]
            def sample(model,params,prompt):
                prompts.append(prompt[:]);return [len(prompts)]*80
            ns={'time':time,'sample_one':sample,'pad_frame_tokens':lambda x:list(x),
                'action_tokens_at':lambda i,*args:[i]*13,'CTX_PREFIX_LEN':1280,'BLOCK':93}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<paper_reference>','exec'),ns)
            context={'prompt_parts':([9]*1280,[8]*93),'sampling_params':None,'num_frames':64,
                     'all_actions':None,'action_ranges':None,'total_frames':66,'device':None}
            ns['run_recap'](None,SimpleNamespace(anchor_frames=1,window_size_w=window,overlap_k=3),context,lambda *a:None)
            state=ReCAPContext([9]*1280,[8]*93,window)
            for step,prompt in enumerate(prompts):
                self.assertEqual(state.prompt(),prompt)
                state.append([step+1]*80,[step+2]*13)
class IntegrityTests(unittest.TestCase):
    def test_corruption_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'asset';p.write_bytes(b'abc')
            entries=[{'path':'asset','bytes':3,'sha256':sha256(p)}]
            self.assertEqual(verify(tmp,entries),[])
            p.write_bytes(b'abd');self.assertTrue(verify(tmp,entries))
            p.unlink();self.assertTrue(verify(tmp,entries))
            with self.assertRaises(ValueError):safe_path(tmp,'../escape')
    def test_configs_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            for p in (ROOT/'configs').glob('*.json'):
                cfg=json.loads(p.read_text());cmd=build_command(cfg,tmp,tmp,Path(tmp)/'new')
                self.assertTrue(Path(cmd[1]).is_file())
                with self.assertRaises(FileExistsError):build_command(cfg,tmp,tmp,tmp)
    def test_protocol_mismatch(self):
        c={'schema_version':1,'backend':'bridge','observed_frames':2,'horizon':32,'arguments':[]}
        with self.assertRaises(ValueError):build_command(c,'.','.','does-not-exist')
if __name__=='__main__':unittest.main()

class DataTests(unittest.TestCase):
    def test_episode_shape_alignment_and_action_mapping(self):
        try:import numpy as np
        except ImportError:self.skipTest('numpy not installed in minimal CPU package')
        from recap.data import validate_episode
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'episode.npz'
            image=np.zeros((3,256,320,3),dtype=np.uint8);action=np.zeros((3,13),dtype=np.float32)
            np.savez(p,image=image,action=action)
            self.assertEqual(validate_episode(p,'calvin',2,1)['frames'],3)
            with self.assertRaises(ValueError):validate_episode(p,'calvin',2,2)
            action[:,8]=1;np.savez(p,image=image,action=action)
            with self.assertRaises(ValueError):validate_episode(p,'bridge',1,2)
            action[:]=float('nan');np.savez(p,image=image,action=action)
            with self.assertRaises(ValueError):validate_episode(p,'rt1',2,1)
