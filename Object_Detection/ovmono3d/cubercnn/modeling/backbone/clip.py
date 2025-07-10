from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
import torch
import torch.nn.functional as F
import einops as E
import unittest
import open_clip
from cubercnn.modeling.backbone.dino import tokens_to_output
from typing import Tuple

class CLIPBackbone(Backbone):
    def __init__(self, cfg, input_shape, arch="ViT-B-16", checkpoint="openai", output="dense", layer=-1, return_multilayer=False, out_feature="last_feat",):
        super().__init__()
        assert output in ["dense-cls", "cls", "gap", "dense"]
        self.output = output
        _clip_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=checkpoint)
        _clip_model = _clip_model.to(torch.float32)
        self.visual = _clip_model.visual
        del _clip_model
        self.patch_size = self.visual.conv1.stride[0]
        feat_dim = self.visual.transformer.width
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim
        feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]
        n_layers = len(self.visual.transformer.resblocks)
        multilayers = [n_layers // 4 - 1, n_layers // 2 - 1, n_layers // 4 * 3 - 1, n_layers - 1]
        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        self._out_feature_channels = {out_feature: feat_dim}
        self._out_feature_strides = {out_feature: self.patch_size}
        self._out_features = [out_feature]

    def forward(self, images):
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)
        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))
        embeds = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break
        outputs = {}
        for i, _x in enumerate(embeds):
            _x = tokens_to_output(self.output, _x[:, 1:], _x[:, 0], out_hw)
            outputs[self._out_features[i]] = _x
        return outputs

def resize_pos_embed(pos_embed: torch.Tensor, hw: Tuple[int, int], has_cls_token: bool = True):
    n_grid = pos_embed.shape[0] - 1 if has_cls_token else pos_embed.shape[0]
    if n_grid == hw[0] * hw[1]:
        return pos_embed
    if has_cls_token:
        cls_embed, pos_embed = pos_embed[[0]], pos_embed[1:]
    orig_dim = int(pos_embed.shape[0] ** 0.5)
    pos_embed = E.rearrange(pos_embed, "(h w) c -> 1 c h w", h=orig_dim)
    pos_embed = F.interpolate(pos_embed, hw, mode="bicubic", align_corners=False, antialias=True)
    pos_embed = E.rearrange(pos_embed, "1 c h w -> (h w) c")
    if has_cls_token:
        pos_embed = torch.cat([cls_embed, pos_embed], dim=0)
    return pos_embed

@BACKBONE_REGISTRY.register()
def build_clip_backbone(cfg, input_shape: ShapeSpec, priors=None):
    arch = cfg.MODEL.CLIP.ARCH
    checkpoint = cfg.MODEL.CLIP.CHECKPOINT
    output = cfg.MODEL.CLIP.OUTPUT
    layer = cfg.MODEL.CLIP.LAYER
    return_multilayer = cfg.MODEL.CLIP.RETURN_MULTILAYER
    bottom_up = CLIPBackbone(cfg, input_shape, arch=arch, checkpoint=checkpoint, output=output, layer=layer, return_multilayer=return_multilayer)
    in_feature = cfg.MODEL.FPN.IN_FEATURE
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    scale_factors = (4.0, 2.0, 1.0, 0.5)
    backbone = SimpleFeaturePyramid(net=bottom_up, in_feature=in_feature, out_channels=out_channels, scale_factors=scale_factors, norm=cfg.MODEL.FPN.NORM, top_block=None, square_pad=cfg.MODEL.FPN.SQUARE_PAD)
    return backbone

class TestCLIPBackbone(unittest.TestCase):
    def setUp(self):
        self.cfg = type('', (), {})()
        self.cfg.MODEL = type('', (), {})()
        self.cfg.MODEL.CLIP = type('', (), {})()
        self.cfg.MODEL.CLIP.ARCH = "ViT-B-16"
        self.cfg.MODEL.CLIP.CHECKPOINT = "openai"
        self.cfg.MODEL.CLIP.OUTPUT = "dense"
        self.cfg.MODEL.CLIP.LAYER = -1
        self.cfg.MODEL.CLIP.RETURN_MULTILAYER = False
        self.cfg.MODEL.FPN = type('', (), {})()
        self.cfg.MODEL.FPN.IN_FEATURE = 'last_feat'
        self.cfg.MODEL.FPN.OUT_CHANNELS = 256
        self.cfg.MODEL.FPN.NORM = "LN"
        self.cfg.MODEL.FPN.FUSE_TYPE = "sum"
        self.cfg.MODEL.FPN.SQUARE_PAD = 512
        self.input_shape = ShapeSpec(channels=3, height=512, width=512)

    def test_clip_backbone_forward(self):
        backbone = build_clip_backbone(self.cfg, self.input_shape)
        x = torch.randn(1, 3, 512, 512)
        outputs = backbone(x)
        print(backbone.net.output_shape())
        for key, output in outputs.items():
            print(key, output.shape)

if __name__ == "__main__":
    unittest.main()