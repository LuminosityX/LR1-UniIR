import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Union
from transformers import AutoModel
import os, sys, json, io, requests, argparse, random, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from PIL import Image
import torch, faiss
from collections import defaultdict
from transformers import AutoModel
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms as T  # ✅ 放在这里
import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Union
from transformers import AutoModel
import os, sys, json, io, requests, argparse, random, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from PIL import Image
import torch, faiss
from collections import defaultdict
from transformers import AutoModel
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict
import sys, os
sys.path.insert(0, "/data/jina-v4-local-copy")  # 保证优先搜到原目录
# import os
# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)

model_dir = Path("/data/jina-v4-local")
#device = "cuda" if torch.cuda.is_available() else "cpu"
#####改了
local_rank = int(os.environ.get("LOCAL_RANK", 0))   # torchrun 会自动设
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
random.seed(42); np.random.seed(42); torch.manual_seed(42)
class JinaV4Tokenizer:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, text, **kwargs):
        # 强制只走文本分支
        return self.processor(text=text, images=None, **kwargs)

    def __getattr__(self, name):
        return getattr(self.processor, name)
import torch
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
from peft import PeftModel, LoraConfig
from transformers import AutoModel
from pathlib import Path
import torch
from custom_lora_module import MultiAdapterLinear
from omegaconf import DictConfig
from typing import Optional
# import os
# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)

model_dir = Path("/data/jina-v4-local-copy")
#device = "cuda" if torch.cuda.is_available() else "cpu"
#####改了
local_rank = int(os.environ.get("LOCAL_RANK", 0))   # torchrun 会自动设
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
random.seed(42); np.random.seed(42); torch.manual_seed(42)
class JinaV4Tokenizer:
    def __init__(self, processor):
        self.processor = processor

    
    def __call__(self, text, **kwargs):
        kwargs.setdefault("padding", True)
        kwargs.setdefault("return_tensors", "pt")
        return BatchEncoding(
            self.processor(text=text, images=None, **kwargs)
        )
        

    def __getattr__(self, name):
        return getattr(self.processor, name)

class JinaV4UniIR(nn.Module):
    def __init__(self, config):
        # super().__init__()
        # self.dummy = nn.Parameter(torch.zeros(1))
        # model_dir = Path("/data/jina-v4-local-copy")
        # self.model = AutoModel.from_pretrained(
        #     str(model_dir), trust_remote_code=True, dtype=torch.float16
        # ).to(device).eval()
        # self.device = device
        # self.model.processor = getattr(self.model, 'processor', None)
        # if self.model.processor is None:
        #     from .modeling_jina_embeddings_v4 import JinaEmbeddingsV4Processor
        #     self.model.processor = JinaEmbeddingsV4Processor.from_pretrained(
        #         str(model_dir),
        #         trust_remote_code=True
        #     )
        #print("[DEBUG] JinaV4UniIR.__init__() called")
        super().__init__()
        self.device = device
        model_dir = Path("/data/jina-v4-local-copy")
        ckpt_path = "/data/LR1-UniIR/checkpoint/jina_v4t/Large/Instruct/InBatch/jina_v4t_step5new.pth"

        # 1. 加载基础模型
        base_model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            dtype=torch.float16,
        )

        # 2. 加载 LoRA 配置
        lora_config = LoraConfig.from_pretrained(str(model_dir / "adapters"))
        lora_config.inference_mode = True
        lora_config._custom_modules = {
            torch.nn.modules.linear.Linear: lambda: MultiAdapterLinear(
                task_names=base_model.config.task_names
            )
        }

        # 3. 构建 PeftModel
        model = PeftModel.from_pretrained(
            model=base_model,
            model_id=str(model_dir / "adapters"),
            config=lora_config,
            is_trainable=False,
        )

        # 4. 加载微调后的 LoRA 权重
        torch.serialization.add_safe_globals([DictConfig, ContainerMetadata])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)

        # 5. 合并 LoRA 权重到基础模型
        self.model = model
        self.model = self.model.to(device).eval()
        print(">>> 权重加载完成")
        # 6. 加载 processor
        from modeling_jina_embeddings_v4 import JinaEmbeddingsV4Processor
        self.model.processor = JinaEmbeddingsV4Processor.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
    def forward(self, batch, encode_mbeir_batch=False, **kwargs):
        
        if encode_mbeir_batch:
            # print(f"[DEBUG] JinaV4UniIR.forward() called with encode_mbeir_batch={encode_mbeir_batch}")
            # print(f"[DEBUG] batch keys: {list(batch.keys())}")
            # # 如果你还想看模态
            # print(f"[DEBUG] modality slice: {batch.get('modality', None)[:2] if 'modality' in batch else None}")
            return self.encode_mbeir_batch(batch)
        # 其他模式可扩展
        raise ValueError("JinaV4UniIR: 未知 forward 模式")
    def eval(self):
        super().eval()
        self.model.eval()   # 让内部 Jina 模型也进 eval 模式
        return self
        # ---------- 新增：Tensor → PIL ----------


    def _tensor_to_pil(self, img_tensor: torch.Tensor) -> Image.Image:
        """
        把 [C, H, W] 的 Tensor（0~1 或 ImageNet 归一化）还原成 PIL.Image
        """
        # 反归一化
        print("ohno暂未删除")
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(img_tensor.device)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(img_tensor.device)
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return transforms.ToPILImage()(img_tensor.cpu())
    
    
    @torch.no_grad()
    def encode_mbeir_batch(self, batch):
        # 1. 反解原始字段
        if "index_mapping" in batch and "query" in batch["index_mapping"]:
            index_mapping = batch["index_mapping"]["query"]
            qid_list = batch.get("qid_list", [])          # query 侧
                     # 候选池侧
            id_list  = qid_list
            # 2. 从 collator 拿回原始 txt / img
            all_txt = batch["txt_batched"]["input_ids"]   # [B, L]
            all_img = batch["image_batched"]              # [B, C, H, W]
            #print("alltxt",all_txt)
            # 1. 不管外边给的是 tensor 还是 list，先变成 [B, L] 的 tensor
            # 直接整批解码，别再遍历
            txt_list = self.model.processor.tokenizer.batch_decode(
                all_txt, skip_special_tokens=True
            )  # -> List[str] 长度 = B
            # 3. 反解 modality（根据 img 是否空）
            #print("txtlist",txt_list)
            modalities, texts, images = [], [], []
                    
            # 1. 重建样本顺序表
            samples = [] 
                # ========== Candidate 路径：顺序遍历 ==========
            for i in range(len(txt_list)):
                txt = txt_list[i]
                img = all_img[i] 
                #if all_img[i].sum() > 0 else None
                mod = ("text" if img is None else
                    "image" if txt == "" else
                    "image,text")
                samples.append((i, mod, txt, img))
            #print("samples",samples)
            # 2. 按模态分组
            grp = defaultdict(list)
            for idx, mod, txt, img in samples:
                grp[mod].append((idx, txt, img))

            # 3. 逐组编码
            emb_dict = {}   # idx -> ndarray
            for mod, items in grp.items():
                idxs, texts, imgs = zip(*items)
                if mod == "text":
                    embs = self._encode_text(list(texts), prompt_name="query", batch_size=len(texts))
                elif mod == "image":
                    embs = self._encode_images(list(imgs), prompt_name="query", batch_size=len(imgs))
                else:  # "image,text"
                    #print("[imgs]",imgs)
                    embs = self._encode_fusion(list(texts), list(imgs), prompt_name="query", batch_size=len(texts))
                for j, idx in enumerate(idxs):
                    emb_dict[idx] = embs[j]   # numpy vector [D]

            # 4. 按原始顺序拼回
            embs_ordered = [emb_dict[i] for i, _, _, _ in samples]
            emb = np.stack(embs_ordered)      # [B, D]
            # print("id_list",id_list)
            print("[DEBUG] query_emb stats:",
                emb.mean().item(),
                emb.std().item())
            # print("emb",emb)
            return torch.from_numpy(emb).float(), id_list
            
            

        else:
            index_mapping = None  # 候选池没有 query 结构
            did_list = batch.get("did_list", [])          # 候选池侧
            id_list  = did_list
            
            # 2. 从 collator 拿回原始 txt / img
            all_txt = batch["txt_batched"]["input_ids"]   # [B, L]
            all_img = batch["image_batched"]              # [B, C, H, W]
            #print("alltxt",all_txt)
            # 1. 不管外边给的是 tensor 还是 list，先变成 [B, L] 的 tensor
            # 直接整批解码，别再遍历
            txt_list = self.model.processor.tokenizer.batch_decode(
                all_txt, skip_special_tokens=True
            )  # -> List[str] 长度 = B
            # 3. 反解 modality（根据 img 是否空）
            #print("txtlist",txt_list)
            modalities, texts, images = [], [], []
                    
            # 1. 重建样本顺序表
            samples = [] 
                # ========== Candidate 路径：顺序遍历 ==========
            for i in range(len(txt_list)):
                txt = txt_list[i]
                img = all_img[i] 
                #if all_img[i].sum() > 0 else None
                mod = ("text" if img is None else
                    "image" if txt == "" else
                    "image,text")
                samples.append((i, mod, txt, img))
            #print("samples",samples)
            # 2. 按模态分组
            grp = defaultdict(list)
            for idx, mod, txt, img in samples:
                grp[mod].append((idx, txt, img))

            # 3. 逐组编码
            emb_dict = {}   # idx -> ndarray
            for mod, items in grp.items():
                idxs, texts, imgs = zip(*items)
                if mod == "text":
                    embs = self._encode_text(list(texts), prompt_name="passage", batch_size=len(texts))
                elif mod == "image":
                    embs = self._encode_images(list(imgs), prompt_name="passage", batch_size=len(imgs))
                else:  # "image,text"
                    embs = self._encode_fusion(list(texts), list(imgs), prompt_name="passage", batch_size=len(texts))
                for j, idx in enumerate(idxs):
                    emb_dict[idx] = embs[j]   # numpy vector [D]

            # 4. 按原始顺序拼回
            embs_ordered = [emb_dict[i] for i, _, _, _ in samples]
            emb = np.stack(embs_ordered)      # [B, D]
            # print("id_list",id_list)
            print("[DEBUG] target_emb stats:",
                emb.mean().item(),
                emb.std().item())
            # print("emb",emb)
            return torch.from_numpy(emb).float(), id_list
            #return emb, id_list
    # def encode_mbeir_batch(self, batch):
    #     """
    #     适配新版 JinaEmbeddingsV4Model.forward
    #     图像已是 PIL.Image，无需转换
    #     明确区分 query 侧 vs candidate 侧
    #     """
    #     image_preprocessed = batch.get("image_preprocessed", False)

    #     # 1. 判断当前 batch 是 query 还是 candidate
    #     is_query = "index_mapping" in batch and "query" in batch["index_mapping"]
    #     side     = "query" if is_query else "candidate"  # prompt_name 用
    #     id_list  = batch.get("qid_list" if is_query else "did_list", [])

    #     # 2. 解析文本、图像
    #     txt_tokens = batch["txt_batched"]["input_ids"]      # [B, L]
    #     txt_mask   = batch["txt_batched"]["attention_mask"]
    #     images     = batch["image_batched"]                 # List[PIL.Image]

    #     texts = self.model.processor.tokenizer.batch_decode(
    #         txt_tokens, skip_special_tokens=True
    #     )

    #     # 3. 模态判断（样本级）
    #     def _modality(t, img):
    #         txt_empty = t.strip() == ""
    #         img_empty = img is None
    #         if img_empty and not txt_empty:
    #             return "text"
    #         if not img_empty and txt_empty:
    #             return "image"
    #         if not img_empty and not txt_empty:
    #             return "image,text"
    #         return "text"

    #     modalities = [_modality(t, img) for t, img in zip(texts, images)]

    #     # 4. 构造输入（统一用 query / target 字段）
    #     #    新版 forward 要求：
    #     #      query_type / target_type / task_label / return_loss=False
    #     #    我们把整个 batch 当作 **一侧** 传进去
    #     mod = modalities[0]          # UniIR 保证同 batch 同模态
    #     if is_query:
    #         # 当前 batch 是 query 侧
    #         output = self.model(
    #             query_type=mod,
    #             target_type="text",        # dummy，必须传
    #             task_label="retrieval",
    #             return_loss=False,
    #             query_input_ids=txt_tokens,
    #             query_attention_mask=txt_mask,
    #             query_pixel_values=images,  # PIL 列表
    #         )
    #         embs = output.query_embeddings          # [B, D]
    #     else:
    #         # 当前 batch 是 candidate 侧
    #         output = self.model(
    #             query_type="text",           # dummy
    #             target_type=mod,
    #             task_label="retrieval",
    #             return_loss=False,
    #             target_input_ids=txt_tokens,
    #             target_attention_mask=txt_mask,
    #             target_pixel_values=images,  # PIL 列表
    #         )
    #         embs = output.target_embeddings         # [B, D]

    #     return embs.cpu().numpy(), id_list
       # ========================== 修改 encode_mbeir_batch ==========================
    # def encode_mbeir_batch(self, batch):
    #     # 1. 先取 flag
    #     image_preprocessed = batch.get("image_preprocessed", False)

    #     # 2. 其余逻辑不变，只把「图像」分支换掉
    #     def _encode_grp(mod, idxs, texts, imgs):
    #         if image_preprocessed:
    #             # if mod == "image":
    #             #     return self.encode_image_tensor(imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             # elif mod == "image,text":
    #             #     return self._encode_fusion_tensor(texts, imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             # else:  # text
    #             #     return self._encode_text(texts, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(texts))
    #             if mod == "image":
    #                 return self._encode_images(imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             elif mod == "image,text":
    #                 return self._encode_fusion(texts, imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             else:
    #                 return self._encode_text(texts, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(texts))
    #         else:
    #             # 老逻辑
    #             if mod == "image":
    #                 return self._encode_images(imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             elif mod == "image,text":
    #                 return self._encode_fusion(texts, imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             else:
    #                 return self._encode_text(texts, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(texts))

    #     index_mapping = batch.get("index_mapping", {})
    #     all_txt = batch["txt_batched"]["input_ids"]
    #     all_img = batch["image_batched"]          # Tensor[B,3,H,W]应该是pil
    #     #print("all_image",all_img)
    #     txt_list = self.model.processor.tokenizer.batch_decode(all_txt, skip_special_tokens=True)

    #     samples = []
    #     for i in range(len(txt_list)):
    #         txt = txt_list[i]
    #         #img = all_img[i] if all_img[i].sum() > 0 else None
    #         img = all_img[i]
    #         mod = ("text" if img is None else
    #                "image" if txt == "" else
    #                "image,text")
    #         samples.append((i, mod, txt, img))

    #     grp = defaultdict(list)
    #     for idx, mod, txt, img in samples:
    #         grp[mod].append((idx, txt, img))

    #     emb_dict = {}
    #     for mod, items in grp.items():
    #         idxs, texts, imgs = zip(*items)
    #         # embs = _encode_grp(mod, idxs, texts, imgs)   # ← 唯一改动点
    #         if image_preprocessed:
    #             if mod == "image":
    #                 embs = self._encode_images(imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             elif mod == "image,text":
    #                 #print("[imgs]",imgs)
    #                 embs = self._encode_fusion(texts, imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             else:
    #                 embs = self._encode_text(list(texts), prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(texts))
                
    #             # if mod == "image":
    #             #     embs = self.encode_image_tensor(imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             # elif mod == "image,text":
    #             #     embs = self._encode_fusion_tensor(texts, imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             # else:  # text
    #             #     embs = self._encode_text(list(texts), prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(texts))
    #         else:
    #             # 老逻辑
    #             if mod == "image":
    #                 embs = self._encode_images(imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             elif mod == "image,text":
    #                 embs = self._encode_fusion(texts, imgs, prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(imgs))
    #             else:
    #                 embs = self._encode_text(list(texts), prompt_name="query" if "query" in index_mapping else "passage",batch_size=len(texts))
    #         print(f"[DEBUG] mod={mod}, len(idxs)={len(idxs)}, embs.shape={embs.shape}")
    #         assert len(embs) == len(idxs), "向量数和索引数不一致"
    #         for j, idx in enumerate(idxs):
    #             emb_dict[idx] = embs[j]

    #     embs_ordered = [emb_dict[i] for i, _, _, _ in samples]
    #     emb = np.stack(embs_ordered)
    #     id_list = batch.get("qid_list") or batch.get("did_list")
    #     return torch.from_numpy(emb).float(), id_list

    
    
    
    def _load(self, path_or_url):
        try:
            if str(path_or_url).startswith("http"):
                response = requests.get(path_or_url, timeout=10)
                img = Image.open(BytesIO(response.content))
            else:
                abs_path = os.path.join("/data/M-BEIR", path_or_url)
                if not os.path.exists(abs_path):
                    print(f"Image not found: {abs_path}")
                    return Image.new("RGB", (224, 224), (0, 0, 0))  # 默认空白图
                img = Image.open(abs_path)
            # >>> 只转 RGB，不 resize <<<
            return img.convert("RGB")
        except Exception as e:
            print(f"Error loading image {path_or_url}: {str(e)}")
            return Image.new("RGB", (224, 224), (0, 0, 0))
    
    @torch.no_grad()
    def _encode_text(self, texts, prompt_name, batch_size):
    #print("_encode_text_encode_text_encode_text_encode_text_encode_text_encode_text_encode_text_encode_text_encode_text")
    #texts = np.asarray(texts)
        embs = []
        hidden_size = self.model.config.text_config.hidden_size
        # print("texts",texts)
        if not texts:
            #return np.empty((0, hidden_size), dtype="float16")
            return BatchEncoding(data={
                "input_ids": torch.empty((0, 0), dtype=torch.long, device=self.device),
                "attention_mask": torch.empty((0, 0), dtype=torch.long, device=self.device)
            })

        batch_texts = texts
        try:
            out = self.model.encode_text(
                batch_texts, task="retrieval", prompt_name=prompt_name
            )
            
            # 处理列表输出（每个元素是[hidden_size]）
            if isinstance(out, list):
                # 将每个1D向量转为2D [1, hidden_size]
                batch_emb = torch.stack(out, dim=0)  # [batch, hidden_size]
                embs.append(batch_emb.cpu().numpy())
            
            # 处理张量输出
            elif isinstance(out, torch.Tensor):
                #print("[out]",out)
                if out.dim() == 1:  # [hidden_size]
                    #print("2222222222222222222")
                    out = out.unsqueeze(0)  # [1, hidden_size]
                embs.append(out.cpu().numpy())
            else:
                raise ValueError(f"Unsupported output type: {type(out)}")
                
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {str(e)}")
            raise
        # del out
        # torch.cuda.empty_cache()
        if not embs:
            return np.empty((0, hidden_size), dtype="float16")
        #print("text",embs)
        return np.concatenate(embs, axis=0)  # [total_samples, hidden_size]
    
    @torch.no_grad()
    def _encode_images(self, images, prompt_name, batch_size):
        # ****** 统一还原成 PIL ******
        print("[images]",images)
        images = [
            self._tensor_to_pil(im) if isinstance(im, torch.Tensor) else
            self._load(im)          if isinstance(im, str)        else
            im                                               # 已是 PIL
            for im in images
        ]
        print("[images]",images)

        embs = []
        for i in range(0, len(images), batch_size):
            out = self.model.encode_image(images[i:i+batch_size], task="retrieval")
            # —— 以下与原逻辑完全一致 ——
            if isinstance(out, list):
                batch_emb = torch.stack(out, dim=0)
            elif hasattr(out, 'single_vec_emb'):
                batch_emb = out.single_vec_emb
            elif isinstance(out, torch.Tensor):
                batch_emb = out.unsqueeze(0) if out.dim() == 1 else out
            else:
                raise ValueError(f"Unsupported output type: {type(out)}")
            embs.append(batch_emb.cpu().numpy())
        if not embs:
            return np.empty((0, self.model.config.text_config.hidden_size), dtype="float16")
        return np.concatenate(embs, axis=0)
    

    # @torch.no_grad()
    # def _encode_fusion(self, texts, images, prompt_name, batch_size):
    #     # ****** 统一还原成 PIL ******
    #     print("[images]",images)
    #     images = [
    #         self._tensor_to_pil(im) if isinstance(im, torch.Tensor) else
    #         self._load(im)          if isinstance(im, str)        else
    #         im
    #         for im in images
    #     ]

    #     embs = []
    #     for i in range(0, len(texts), batch_size):
    #         txt_b = texts[i:i+batch_size]
    #         img_b = images[i:i+batch_size]
    #         prompts = [f"<|im_start|>user\n{t}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n" for t in txt_b]

    #         model_inputs = self.model.processor(
    #             text=prompts,
    #             images=img_b,
    #             padding="longest",
    #             return_tensors="pt",
    #         ).to(self.device)

    #         # —— 以下与原逻辑完全一致 ——
    #         offsets = model_inputs["image_grid_thw"][:, 1] * model_inputs["image_grid_thw"][:, 2]
    #         pixel_values = torch.split(model_inputs["pixel_values"], offsets.tolist())
    #         max_len = max(pv.shape[0] for pv in pixel_values)
    #         pixel_values = [
    #             torch.cat([pv, torch.zeros((max_len - pv.shape[0], pv.shape[1]), dtype=pv.dtype, device=pv.device)])
    #             for pv in pixel_values
    #         ]
    #         model_inputs["pixel_values"] = torch.stack(pixel_values)

    #         # out = self.model(task_label="retrieval", **model_inputs)
    #         out = self.model.forward_original(
    #             task_label="retrieval",
    #             input_ids=model_inputs["input_ids"],
    #             attention_mask=model_inputs["attention_mask"],
    #             #pixel_values=model_inputs["pixel_values"],
    #             image_grid_thw=model_inputs["image_grid_thw"],
    #         )
    #         if hasattr(out, 'single_vec_emb'):
    #             emb = out.single_vec_emb.cpu().numpy()
    #         elif hasattr(out, 'last_hidden_state'):
    #             emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
    #         else:
    #             raise ValueError("Model output missing valid embedding")
    #         embs.append(emb)
    #     if not embs:
    #         return np.empty((0, self.model.config.text_config.hidden_size), dtype="float16")
    #     return np.concatenate(embs, axis=0)
    @torch.no_grad()
    def _encode_fusion(self, texts, images, prompt_name, batch_size):
        # ****** 统一还原成 PIL ******
        images = [
            self._tensor_to_pil(im) if isinstance(im, torch.Tensor) else
            self._load(im)          if isinstance(im, str)        else
            im
            for im in images
        ]

        embs = []
        for i in range(0, len(texts), batch_size):
            txt_b = texts[i:i+batch_size]
            img_b = images[i:i+batch_size]
            prompts = [f"<|im_start|>user\n{t}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n" for t in txt_b]

            model_inputs = self.model.processor(
                text=prompts,
                images=img_b,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)

            # —— 以下与原逻辑完全一致 ——
            offsets = model_inputs["image_grid_thw"][:, 1] * model_inputs["image_grid_thw"][:, 2]
            pixel_values = torch.split(model_inputs["pixel_values"], offsets.tolist())
            max_len = max(pv.shape[0] for pv in pixel_values)
            pixel_values = [
                torch.cat([pv, torch.zeros((max_len - pv.shape[0], pv.shape[1]), dtype=pv.dtype, device=pv.device)])
                for pv in pixel_values
            ]
            model_inputs["pixel_values"] = torch.stack(pixel_values)

            out = self.model.forward_original(task_label="retrieval", **model_inputs)
            if hasattr(out, 'single_vec_emb'):
                emb = out.single_vec_emb.cpu().numpy()
            elif hasattr(out, 'last_hidden_state'):
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            else:
                raise ValueError("Model output missing valid embedding")
            embs.append(emb)
        if not embs:
            return np.empty((0, self.model.config.text_config.hidden_size), dtype="float16")
        return np.concatenate(embs, axis=0)


    def _convert_to_model_format(self, pil_img):
        """将PIL图像转换为模型期望的[3, 2, 14, 14]形状张量"""
        # 1. 调整大小为224x224
        pil_img = pil_img.resize((224, 224), Image.BICUBIC)
        
        # 2. 转换为numpy数组并归一化
        img_array = np.array(pil_img) / 255.0
        img_array = (img_array - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        
        # 3. 转换为模型期望的形状 [3, 2, 14, 14]
        # 这里需要根据模型具体要求实现具体的转换逻辑
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]
        
        # 模拟模型期望的patch嵌入
        # 将224x224图像分割为16x16的patch (224/16=14)
        patches = img_tensor.unfold(1, 16, 16).unfold(2, 16, 16)  # [C, 14, 14, 16, 16]
        
        # 将patch平均分成两组 (2, 14, 14)
        # 这里简化处理，实际可能需要更复杂的划分逻辑
        group1 = patches[:, :, :, :8, :].mean(dim=(3, 4))  # [C, 14, 14]
        group2 = patches[:, :, :, 8:, :].mean(dim=(3, 4))   # [C, 14, 14]
        
        # 组合成最终形状 [3, 2, 14, 14]
        final_tensor = torch.stack([group1, group2], dim=1)
        
        return final_tensor.float()
    def get_img_preprocess_fn(self):
        return None  # 你已经在内部处理了

    def get_tokenizer(self):
        return JinaV4Tokenizer(self.model.processor)
