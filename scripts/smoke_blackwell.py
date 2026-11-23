"""Bounded, offline tests; no pretrained models or pipeline batches."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib
import numpy as np
import torch

def test_torch():
    assert torch.__version__ == '2.7.1+cu128', torch.__version__
    assert torch.cuda.get_device_capability() == (12, 0)
    assert torch.cuda.is_bf16_supported()
    a = torch.arange(256, device='cuda', dtype=torch.float32).reshape(16, 16) / 256
    for dtype in (torch.float32, torch.bfloat16):
        y = a.to(dtype) @ a.to(dtype).T
        torch.testing.assert_close(y.float(), (a @ a.T), rtol=0.01, atol=0.01)
    torch.cuda.synchronize()
    print('PASS torch matmul FP32/BF16', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), flush=True)

def test_knn():
    from simple_knn._C import distCUDA2
    import simple_knn._C as knn_native
    assert Path(knn_native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    print('knn native',knn_native.__file__,flush=True)
    p = torch.tensor([[0,0,0],[1,0,0],[0,2,0],[0,0,3],[2,3,4]], device='cuda', dtype=torch.float32)
    d = torch.cdist(p, p).square()
    d.fill_diagonal_(float('inf'))
    expected = d.topk(3, largest=False).values.mean(1)
    actual = distCUDA2(p)
    torch.testing.assert_close(actual, expected)
    torch.cuda.synchronize()
    print('PASS simple-knn GPU vs brute force', actual.tolist(), flush=True)

def test_gaussian():
    import diff_gaussian_rasterization as dgr
    print('rasterizer source', dgr.__file__, 'native', dgr._C.__file__, flush=True)
    assert 'confidence' not in dgr.__file__
    assert Path(dgr._C.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    def tensor(x): return torch.tensor(x, device='cuda', dtype=torch.float32, requires_grad=True)
    pos = tensor([[0,0,2],[0.2,0.1,2.5]])
    screen = torch.zeros_like(pos, requires_grad=True)
    colors = tensor([[1,0.2,0.1],[0.1,0.3,1]])
    opacity = tensor([[0.7],[0.6]])
    scales = tensor([[0.1]*3,[0.15]*3])
    rotation = tensor([[1,0,0,0],[1,0,0,0]])
    view = torch.eye(4, device='cuda')
    from utils.graphics import getProjectionMatrix
    proj = getProjectionMatrix(0.01, 100.0, 1.5707963267948966, 1.5707963267948966).T.cuda()
    settings = dgr.GaussianRasterizationSettings(32,32,1.,1.,torch.zeros(3,device='cuda'),1.,view,proj,0,torch.zeros(3,device='cuda'),False,False)
    color, radii, depth, alpha = dgr.GaussianRasterizer(settings)(pos,screen,opacity,colors_precomp=colors,scales=scales,rotations=rotation)
    assert (radii > 0).all() and color.max() > 0 and alpha.max() > 0
    assert color.shape == (3,32,32)
    for x in (color, depth, alpha): assert torch.isfinite(x).all()
    (color.square().mean() + depth.mean()*0.001 + alpha.mean()*0.01).backward()
    for name,x in [('positions',pos),('screen',screen),('colors',colors),('opacity',opacity),('scales',scales),('rotation',rotation)]:
        assert x.grad is not None and torch.isfinite(x.grad).all(), name
    assert colors.grad.abs().sum() > 0 and pos.grad.abs().sum() > 0
    torch.cuda.synchronize()
    print('PASS custom gaussian four-output forward/backward', radii.tolist(), 'color sum', color.sum().item(), flush=True)

def test_attention():
    assert importlib.util.find_spec('xformers') is None, 'The verified Blackwell recipe excludes xformers'
    from src.Depth_Anything_V2.depth_anything_v2.dinov2_layers import attention as dino
    assert not dino.XFORMERS_AVAILABLE
    net = dino.MemEffAttention(dim=128,num_heads=4).cuda().eval()
    x = torch.randn(1,16,128,device='cuda',requires_grad=True)
    y = net(x)
    q,k,v = net.qkv(x).reshape(1,16,3,4,32).permute(2,0,3,1,4).unbind(0)
    ref = net.proj(torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(1,16,128))
    torch.testing.assert_close(y,ref,rtol=1e-4,atol=1e-4)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    print('PASS project DINO normal-Tensor fallback forward/backward vs SDPA',flush=True)
    sys.path.insert(0,str(ROOT/'pasd'))
    from pasd.myutils.vaehook import attn_forward_new_pt2_0
    from diffusers.models.attention_processor import Attention
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        net = Attention(query_dim=32,heads=4,dim_head=8,bias=True).cuda().to(dtype).eval()
        x = torch.randn(1,32,4,4,device='cuda',dtype=dtype,requires_grad=True)
        y = attn_forward_new_pt2_0(net,x)
        torch.testing.assert_close(y,net(x),rtol=0.02,atol=0.02)
        y.float().square().mean().backward()
        assert torch.isfinite(x.grad).all()
        torch.cuda.synchronize()
        print('PASS project PASD SDPA forward/backward',dtype,flush=True)

def test_depth():
    from instaOmniDepth import depthmapAlign as da
    from EigenSolvers import LinearSolver
    import EigenSolvers
    for native in (da,EigenSolvers):
        print('depth native',native.__file__,flush=True)
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    from scipy.sparse import csc_matrix
    a = np.arange(12,dtype=np.float64).reshape(3,4)
    np.testing.assert_array_equal(da.python_numpy_mat_test(a), a)
    m = csc_matrix(np.array([[4.,1.],[1.,3.]]))
    b = np.array([1.,2.])
    x = LinearSolver(m,LinearSolver.SimplicialLDLT).solve(b)
    np.testing.assert_allclose(m @ x,b,rtol=1e-10)
    assert da.init('group') == 0
    assert da.ceres_solver_option(2,50,100,1) == 1
    yy,xx=np.mgrid[1:15,1:15]
    corr=np.column_stack([yy.ravel(),xx.ravel(),yy.ravel(),xx.ravel()]).astype(np.float64)
    base=np.linspace(1.,3.,256).reshape(16,16)
    maps=[base*1.2+0.1,base.copy()]
    before=np.mean(np.abs(maps[0]-maps[1]))
    aligned,coeff=da.depthmap_stitch('',[1.,0.1,0.0001],maps,[0,1],1,{0:{1:corr},1:{0:corr}},2,2,0,0,[],[],0)
    assert len(aligned)==2 and len(coeff)==4
    for item in aligned+coeff: assert np.isfinite(item).all()
    after=np.mean(np.abs(aligned[0]-aligned[1]))
    assert after < before * 0.5, (before,after)
    da.shutdown()
    print('PASS original instaOmniDepth Ceres optimization', 'MAE before',before,'after',after,flush=True)

def test_imports():
    modules=['src.pipeline_flux','src.pipeline_flux_fill','src.Infusion.depth_inpainting.inference.depth_inpainting_pipeline_half','src.Depth_Anything_V2.depth_anything_v2.dpt','utils.lama','utils.depth_alignment','gaussian_renderer','scene','LayerPano','transformers','segment_anything']
    for module in modules:
        importlib.import_module(module)
        print('PASS import',module,flush=True)
    sys.path.insert(0,str(ROOT/'pasd'))
    for module in ['pasd.pipelines.pipeline_pasd','pasd.models.pasd.unet_2d_condition','pasd.models.pasd.controlnet','pasd.myutils.vaehook','open_clip']:
        importlib.import_module(module)
        print('PASS import optional SR',module,flush=True)
    from utility.depthmap_utils import depthmap_pyramid
    pyramid = depthmap_pyramid([np.ones((16,16))], 2, 2)
    assert pyramid[0][0].shape == (8,8) and pyramid[1][0].shape == (16,16)
    np.testing.assert_allclose(pyramid[0][0], 1.)
    from unittest.mock import patch
    from types import SimpleNamespace
    from pasd.myutils import vaehook
    with patch.object(vaehook, 'is_xformers_available', return_value=False), patch.object(vaehook, 'attn_forward_new_pt2_0', return_value='sdpa-route') as fallback:
        queue=[]
        vaehook.attn2task(queue, SimpleNamespace(group_norm=lambda x:x))
        assert queue[2][1](torch.zeros(1)) == 'sdpa-route'
        fallback.assert_called_once()
    print('PASS scalar depth pyramid and PASD no-xformers fallback routing',flush=True)

if __name__ == '__main__':
    torch.manual_seed(0)
    tests={'torch':test_torch,'knn':test_knn,'gaussian':test_gaussian,'attention':test_attention,'depth':test_depth,'imports':test_imports}
    for name in sys.argv[1:] or tests:
        tests[name]()
    print('ALL REQUESTED SMOKE TESTS PASSED',flush=True)
