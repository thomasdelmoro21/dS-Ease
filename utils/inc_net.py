import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, EaseCosineLinear, SimpleContinualLinear
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    elif '_memo' in name:
        if args["model_name"] == "memo":
            from backbone import vit_memo
            _basenet, _adaptive_net = timm.create_model("vit_base_patch16_224_memo", pretrained=True, num_classes=0)
            _basenet.out_dim = 768
            _adaptive_net.out_dim = 768
            return _basenet, _adaptive_net
    # SSF 
    elif '_ssf' in name:
        if args["model_name"] == "aper_ssf"  or args["model_name"] == "ranpac" or args["model_name"] == "fecam":
            from backbone import vit_ssf
            if name == "pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_ssf":
                model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"] == "aper_vpt"  or args["model_name"] == "ranpac" or args["model_name"] == "fecam":
            from backbone.vpt import build_promptmodel
            if name == "pretrained_vit_b16_224_vpt":
                basicmodelname = "vit_base_patch16_224" 
            elif name == "pretrained_vit_b16_224_in21k_vpt":
                basicmodelname = "vit_base_patch16_224_in21k"
            
            print("modelname,", name, "basicmodelname", basicmodelname)
            VPT_type = "Deep"
            if args["vpt_type"] == 'shallow':
                VPT_type = "Shallow"
            Prompt_Token_num = args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "aper_adapter" or args["model_name"] == "ranpac" or args["model_name"] == "fecam":
            from backbone import vit_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # L2P
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            from backbone import vit_l2p
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # dualprompt
    elif '_dualprompt' in name:
        if args["model_name"] == "dualprompt":
            from backbone import vit_dualprompt
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
                use_g_prompt=args["use_g_prompt"],
                g_prompt_length=args["g_prompt_length"],
                g_prompt_layer_idx=args["g_prompt_layer_idx"],
                use_prefix_tune_for_g_prompt=args["use_prefix_tune_for_g_prompt"],
                use_e_prompt=args["use_e_prompt"],
                e_prompt_layer_idx=args["e_prompt_layer_idx"],
                use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_e_prompt"],
                same_key_value=args["same_key_value"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # Coda_Prompt
    elif '_coda_prompt' in name:
        if args["model_name"] == "coda_prompt":
            from backbone import vit_coda_promtpt
            model = timm.create_model(args["backbone_type"], pretrained=args["pretrained"])
            # model = vision_transformer_coda_prompt.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
            #                 num_heads=12, ckpt_layer=0,
            #                 drop_path_rate=0)
            # from timm.models import vit_base_patch16_224
            # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # del load_dict['head.weight']; del load_dict['head.bias']
            # model.load_state_dict(load_dict)
            return model
    elif '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease" or args["model_name"] == "dsease" or args["model_name"] == "dsease_hoc" or args["model_name"] == "psrd_ease":
            from backbone import vit_ease
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_ease":
                model = vit_ease.vit_base_patch16_224_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_ease":
                model = vit_ease.vit_base_patch16_224_in21k_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif '_lae' in name:
        from backbone import vit_lae
        model = timm.create_model(args["backbone_type"], pretrained=True)
        return model
        
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes):
        self._cur_task += 1
        
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[ : old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        fc = EaseCosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
            
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class SimplexEaseNet(BaseNet):
    def __init__(self, args, pretrained=True, total_cls=100):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

        # D-SIMPLEX
        self.total_cls = total_cls

        self.reweight = args["concat_reweight"]
        self.junction_list = nn.ModuleList()
        self.add_new_junction(self._device)
        self.dsimplex_layer = nn.Linear(self.total_cls - 1, self.total_cls, bias=False)

        fixed_weights = self.dsimplex()
        self.dsimplex_layer.weight.requires_grad = False
        self.dsimplex_layer.weight.copy_(fixed_weights)
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)
    
    def extract_vector(self, x):
        return self.backbone(x)

    def dsimplex(self, device='cuda'):
        def simplex_coordinates(n, device):
            t = torch.zeros((n + 1, n), device=device)
            torch.eye(n, out=t[:-1,:], device=device)
            val = (1.0 - torch.sqrt(1.0 + torch.tensor([n], device=device))) / n
            t[-1,:].add_(val)
            t.add_(-torch.mean(t, dim=0))
            t.div_(torch.norm(t, p=2, dim=1, keepdim=True)+ 1e-8)
            return t

        ds = simplex_coordinates(self.total_cls - 1, device)
        return ds

    def add_new_junction(self, device):
        self.junction_list.append(nn.Linear(self.out_dim, self.total_cls - 1, bias=False).requires_grad_(True))

    def update_junctions(self, device):
        self._cur_task += 1

        if self._cur_task > 0:
            self.junction_list[self._cur_task - 1].requires_grad_(False)
            self.add_new_junction(device)

    def forward(self, x, test=False):
        if test == False:
            vit_out = self.backbone.forward(x, False)
            out = self.junction_list[self._cur_task](vit_out)
            out = self.dsimplex_layer(out)
        else:
            vit_out = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm, use_dsimplex=True)
            if not self.reweight:
                features = []
                for x, junction in zip(vit_out, self.junction_list):
                    features.append(junction(x))
                out = torch.mean(torch.stack(features), dim=0)
                out = self.dsimplex_layer(out)
            else:
                features = []
                for x, junction in zip(vit_out, self.junction_list):
                    features.append(junction(x).unsqueeze(1))
                out = torch.cat(features, dim=1)
                out = self.dsimplex_layer(out)
                out = self.dsimplex_reweight(out)
            
        out = {'logits': out}
        out.update({"features": vit_out})
        return out

    def dsimplex_reweight(self, x):
        x = x.permute(1, 2, 0)
        for i, adapt in enumerate(x):
            for j, cls in enumerate(adapt):
                if j >= i * self.inc and j < (i + 1) * self.inc:
                    pass
                else:
                    x[i, j] = cls * self.alpha
        x = x.permute(2, 0, 1)
        return torch.mean(x, dim=1)

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class HocSimplexEaseNet(BaseNet):
    def __init__(self, args, pretrained=True, total_cls=100):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

        # D-SIMPLEX
        self.total_cls = total_cls

        self.junction = nn.Linear(self.out_dim, self.total_cls - 1, bias = False)
        self.dsimplex_layer = nn.Linear(self.total_cls - 1, self.total_cls, bias=False)

        fixed_weights = self.dsimplex()
        self.dsimplex_layer.weight.requires_grad = False
        self.dsimplex_layer.weight.copy_(fixed_weights)

        self.proxy_junction = nn.Linear(self.out_dim, self.total_cls - 1, bias=False)
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)
    
    def extract_vector(self, x):
        return self.backbone(x)

    def dsimplex(self, device='cuda'):
        def simplex_coordinates(n, device):
            t = torch.zeros((n + 1, n), device=device)
            torch.eye(n, out=t[:-1,:], device=device)
            val = (1.0 - torch.sqrt(1.0 + torch.tensor([n], device=device))) / n
            t[-1,:].add_(val)
            t.add_(-torch.mean(t, dim=0))
            t.div_(torch.norm(t, p=2, dim=1, keepdim=True)+ 1e-8)
            return t

        ds = simplex_coordinates(self.total_cls - 1, device)
        return ds

    def expand_junction(self):
        self._cur_task += 1

        self.proxy_junction.reset_parameters()

        if self._cur_task > 0:
            new_junction = nn.Linear(self.out_dim * (self._cur_task + 1), self.total_cls - 1, bias=False)
            with torch.no_grad():
                new_junction.weight[:, :self.junction.in_features] = self.junction.weight
                self.junction = new_junction
    
    def update_junctions(self):
        with torch.no_grad():
            self.junction.weight[:, -self.out_dim:] = self.proxy_junction.weight

    def forward(self, x, hoc=False):
        if hoc == False:  # Train the Task Adapter
            vit_out = self.backbone.forward(x, test=False, use_init_ptm=self.use_init_ptm, use_dsimplex=True)  # test=True to obtain features from all adapters, vit_out: type = list of tensors
            features = self.proxy_junction(vit_out)  # Proxy Junction to train adapter
            out = self.dsimplex_layer(features)
        else:  # Train Junction Layer with HOC Loss  /  Test
            with torch.no_grad():    
                vit_out = self.backbone.forward(x, test=True, use_init_ptm=self.use_init_ptm, use_dsimplex=True)  # vit_out: type = list of tensors
            vit_out = torch.cat(vit_out, dim=1)  # vit_out: type = tensor
            features = self.junction(vit_out)
            out = self.dsimplex_layer(features)

        out = {'logits': out}
        out.update({"features": features})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class PSRDEaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

        self.head = ProjectionMLP(self.out_dim, self.args["head_hidden_dim"], self.args["head_proj_dim"], batch_norm=self.args["head_batch_norm"], num_layers=self.args["head_num_layers"])
    
        self.prototypes = Prototypes(
            feat_dim=self.out_dim,
            n_classes_per_task=self.inc,
            n_tasks=self.args["num_tasks"],
            half_iid=self.args["half_iid"],
        )

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        out = dict()
        if test == False:
            features = self.backbone.forward(x, False)
            proj_features = self.head(features)
        else:
            features = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            proj_features = self.head(features)

        out.update({"features": features, "proj_features": proj_features})   
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class ProjectionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        super(ProjectionMLP, self).__init__()

        self.layers = self._make_layers(
            dim_in, hidden_dim, feat_dim, batch_norm, num_layers
        )

    def _make_layers(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        layers = []
        layers.append(add_linear(dim_in, hidden_dim, batch_norm=batch_norm, relu=True))

        for _ in range(num_layers - 2):
            layers.append(
                add_linear(hidden_dim, hidden_dim, batch_norm=batch_norm, relu=True)
            )

        layers.append(add_linear(hidden_dim, feat_dim, batch_norm=False, relu=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def add_linear(dim_in, dim_out, batch_norm, relu):
    layers = []
    layers.append(nn.Linear(dim_in, dim_out))
    if batch_norm:
        layers.append(nn.BatchNorm1d(dim_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class Prototypes(nn.Module):
        def __init__(
            self,
            feat_dim: int,
            n_classes_per_task: int,
            n_tasks: int,
            half_iid: bool = False,
        ):
            super(Prototypes, self).__init__()

            self.heads = self._create_prototypes(
                dim_in=feat_dim,
                n_classes=n_classes_per_task,
                n_heads=n_tasks,
                half_iid=half_iid,
            )

        def _create_prototypes(
            self, dim_in: int, n_classes: int, n_heads: int, half_iid: bool = False
        ) -> torch.nn.ModuleDict:

            first_head_id = 0
            if half_iid:
                first_head_id = (n_heads // 2) - 1
                first_head_n_classes = n_classes * (n_heads // 2)

            layers = {}
            for t in range(first_head_id, n_heads):

                if half_iid and (t == first_head_id):
                    layers[str(t)] = nn.Linear(dim_in, first_head_n_classes, bias=False)
                else:
                    layers[str(t)] = nn.Linear(dim_in, n_classes, bias=False)

            return nn.ModuleDict(layers)

        def forward(self, x: torch.FloatTensor, task_id: int) -> torch.FloatTensor:
            out = self.heads[str(task_id)](x)
            return out