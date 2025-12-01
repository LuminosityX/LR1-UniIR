# Standard library
import argparse
import logging
import os
import random
import time
import datetime
from PIL import Image
from transformers import AutoModel
# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
### 混合精度（AMP）下的梯度缩放，避免下溢
from torch.cuda.amp import GradScaler
### 余弦退火学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR
### 加载/操作 YAML 风格配置
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb
import swanlab, os
from peft import PeftModel
# 关键：让 swanlab 接管 wandb 的日志，并缓存在本地
swanlab.sync_wandb()   # ① 离线双写
logging.getLogger("swanlab").setLevel(logging.WARNING)
os.environ["WANDB_MODE"] = "offline"  # ② 让 wandb 也写本地，不撞墙
from pathlib import Path
# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    Mode,
    JinaV4Collator,

)
# from models.uniir_blip.backbone.blip import load_checkpoint
# from models.uniir_blip.blip_featurefusion.blip_ff import blip_ff
# from models.uniir_blip.blip_scorefusion.blip_sf import blip_sf
# from models.uniir_blip.engine import train_one_epoch, eval_engine
# import models.uniir_blip.utils as utils

import sys
sys.path.insert(0, "/data/jina-v4-local-copy")   # 让 python 找到 jina 文件

from modeling_jina_embeddings_v4 import JinaEmbeddingsV4Model, JinaEmbeddingsV4Processor
from engine import train_one_epoch, eval_engine   # 你前面已改好的文件
import utils 

# Set up logger
logger = logging.getLogger()

# from debug_jina_v4 import apply_debug
# apply_debug(JinaEmbeddingsV4Model, JinaV4Collator)   # 替换成真实类名

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
#     ckpt_config = config.model.ckpt_config
#     model_name = config.model.short_name.lower()
#     checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
#     save_obj = {
#         "model": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "scheduler": scheduler.state_dict(),
#         "config": config,
#         "epoch": epoch,
#         "scaler": scaler.state_dict(),
#     }
#     checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
#     os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
#     torch.save(save_obj, checkpoint_path)
#     print(f"Saved checkpoint to {checkpoint_path}")

def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config, suffix=None):
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    # 如果传了 suffix，就用 suffix，否则按原来 epoch 命名
    if suffix is None:
        checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    else:
        checkpoint_name = f"{model_name}_{suffix}.pth"

    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats


def train(
    train_loader,
    val_loader,
    model,
    model_without_ddp,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch,
):
    gpu_id = config.dist_config.gpu_id
    is_distributed_mode = config.dist_config.distributed_mode
    ### best_inbatch_accuracy 用于追踪最优 in-batch acc。
    global_step, total_loss, best_inbatch_accuracy = (
        0,
        0.0,
        0.0,
    )  # TODO: global_step is not used.
    best_epoch = 0
    model.zero_grad()

    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Set different seed for different epoch
        ### DDP 关键点：每个 epoch 为 DistributedSampler 设置新种子，确保各 rank 的 shuffle 一致。
        
        
        if is_distributed_mode:
            train_loader.sampler.set_epoch(epoch)
        global_step = 0
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            gpu_id,
            scheduler,
            global_step,
            scaler,
            config,
        )

        eval_freq = config.evaluator.eval_freq
        ### 若无 val_loader（评估关闭）或未到评估频次，仅记录 train 结果并仍然保存 checkpoint（保障每个 epoch 的存档）。
        if val_loader is None or epoch % eval_freq != 0:
            log_stats = log_results(train_stats, None, None, epoch, best_epoch)
            ### 只在主进程保存，避免多进程竞争文件。
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
        else:
            ### eval_engine 负责评估，返回字典，代码期望有 "inbatch_accuracy"
            val_status = eval_engine(model_without_ddp, model, val_loader, gpu_id, config)
            try:
                inbatch_accuracy = float(val_status["inbatch_accuracy"])
            except ValueError:
                print(f"Error: Expected a number but got '{val_status['inbatch_accuracy']}'")
                inbatch_accuracy = 100.0
            # Note: still save the model even if the in-batch accuracy is not the best
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
            if inbatch_accuracy >= best_inbatch_accuracy:
                # if utils.is_main_process():
                #     save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
                best_inbatch_accuracy = inbatch_accuracy
                best_epoch = epoch
            log_stats = log_results(train_stats, val_status, None, epoch, best_epoch)

        if utils.is_main_process():
            # logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
            # logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
            # with open(logger_out_path, "a") as f:
            #     f.write(json.dumps(log_stats) + "\n")
            ### 仅主进程往 WandB 推指标，避免重复
            if config.wandb_config.enabled:
                wandb.log(log_stats)
            # print(f'Epoch {epoch}, Loss: {loss.item()}')
        ### barrier：同步所有进程，确保主进程写操作完成。
	    ### empty_cache：释放未使用的显存缓存，缓解碎片问题（非必须，但长跑稳定性更好）。
        # 在 optimizer.step() 之后立刻打
        
        dist.barrier()  # Wait for the master process to finish writing the log file
        torch.cuda.empty_cache()


def main(config):
    is_distributed_mode = config.dist_config.distributed_mode

    # Set up seed for reproducibility
    seed = config.seed + utils.get_rank()
    set_seed(seed)

    ### 打开 cuDNN benchmark，固定输入尺寸可提升性能；若输入尺寸波动较大，会反而降低稳定性
    cudnn.benchmark = True

    #### 换成Jina V4
    # Initialize and load model
    print("Creating jina model...")
    model_config = config.model
    ckpt_config = model_config.ckpt_config
    if model_config.name == "JinaEmbeddingsV4":
        import torch
        from pathlib import Path
        from transformers import AutoModel
        from peft import PeftModel

        model_dir = Path("/data/jina-v4-local-copy")
        ADAPTER_PATH_ROUND1 = Path("/data/jina-v4-local-copy/adapters_exp0329")  # ← 第一次 LoRA

        # 1) 裸载入 base
        base_model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # 2) 挂上已有 adapter，名字用 default（与文件夹里保存的一致）
        # 1. 先挂 retrieval
        model = PeftModel.from_pretrained(base_model,
                                        "/data/jina-v4-local-copy/adapters_exp0329",
                                        adapter_name="retrieval")

        # 2. 继续挂另外两个（PEFT 0.7.0+ 支持多 adapter）
        model.load_adapter("/data/jina-v4-local-copy/adapters_exp0329",
                        adapter_name="text-matching")
        model.load_adapter("/data/jina-v4-local-copy/adapters_exp0329",
                        adapter_name="code")
        from safetensors import safe_open
        import numpy as np

        # st_file = "/data/jina-v4-local-copy/adapters_exp0329/adapter_model.safetensors"

        # with safe_open(st_file, framework="pt", device="cpu") as f:
        #     for k in f.keys():
        #         if "lora_B" in k:
        #             tensor = f.get_tensor(k)          # numpy array
        #             print(k, tensor.shape)
        #             print("max =", tensor.max(), "mean =", tensor.mean())
        import torch
        for n, p in model.named_parameters():
            if "lora_B" in n and "default" in n:
                print(n, p.max().item(), p.mean().item())
                break
        # 3) 接着 default 继续训练：打开梯度
        model.set_adapter("default")
        model.enable_adapter_layers()   # 让 A/B 可训练

        # 4) 你原来的温度/统计逻辑保持不动
        model.temperature.data = torch.tensor(0.07, device=model.temperature.device)
        print("[DEBUG] temperature reset to:", model.temperature.item())

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print("Active adapter:", model.active_adapters)
    # if model_config.name == "JinaEmbeddingsV4":
    #     model_dir = Path("/data/jina-v4-local-copy")
    #     model = AutoModel.from_pretrained(
    #         str(model_dir), trust_remote_code=True, dtype=torch.float16,is_training=True,
    #     )
    #     # 在 main() 里，加载完模型后、DDP 前
    #     model.temperature.data = torch.tensor(0.07, device=model.temperature.device)
    #     print("[DEBUG] temperature reset to:", model.temperature.item())
    #     total_params = sum(p.numel() for p in model.parameters())
    #     # 加载后立刻看有哪些 adapter
    #     print("Available adapters:", list(model.peft_config.keys()))
    #     # 用实际存在的名字
    #     model.set_adapter(list(model.peft_config.keys())[0])   # 先用第一个
    #     print("Active adapter:", model.active_adapters)
    #     # 计算可训练参数的数量
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #     # 输出
    #     print(f"Total parameters: {total_params}")
    #     print(f"Trainable parameters: {trainable_params}")
    # if model_config.name == "BLIPScoreFusion":
    #     model = blip_sf(
    #         pretrained=ckpt_config.pretrained_blip_url,  # This always saved to cache
    #         image_size=model_config.image_size,
    #         vit=model_config.vit,
    #         vit_grad_ckpt=model_config.vit_grad_ckpt,
    #         vit_ckpt_layer=model_config.vit_ckpt_layer,
    #         embed_dim=model_config.embed_dim,
    #         queue_size=model_config.queue_size,
    #         config=model_config,
    #     )
    # elif model_config.name == "BLIPFeatureFusion":
    #     model = blip_ff(
    #         pretrained=ckpt_config.pretrained_blip_url,  # This always saved to cache
    #         ### 224
    #         image_size=model_config.image_size,
    #         ### large
    #         vit=model_config.vit,
    #         ### True，开启梯度检查点，显著节省显存（以算力换显存），训练略变慢。
    #         vit_grad_ckpt=model_config.vit_grad_ckpt,
    #         ### 对 ViT 前 12 层或指定段落启用检查点（具体取决于实现）。可微调该值来平衡显存/速度。
    #         vit_ckpt_layer=model_config.vit_ckpt_layer,
    #         ### 768
    #         embed_dim=model_config.embed_dim,
    #         ### 57960
    #         queue_size=model_config.queue_size,
    #         config=model_config,
    #     )
    
    else:
        raise NotImplementedError(f"Model {config.model} not implemented")

    # Set up optimizer, and scaler
    trainer_config = config.trainer_config
    ############################
    def get_param_groups(model, wd=0.0):
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:          # LoRA 层
                no_decay.append(p)
            else:
                decay.append(p)
        return [{"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]
    # optimizer = torch.optim.AdamW(
    #     params=model.parameters(),
    #     lr=trainer_config.init_lr,
    #     weight_decay=trainer_config.weight_decay,
    # )
    print("[name, p.shape]=====================")
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.shape)
    for n,p in model.named_parameters(): 
        if p.requires_grad:
            if 'lora' in n: 
                print(n, p.max().item(), p.mean().item())
            
    optimizer = torch.optim.AdamW(get_param_groups(model), lr=trainer_config.init_lr)
    scaler = GradScaler()  # Initialize the GradScaler

    #### 换了模型，需要考虑修改这里的代码
    # If resume training, load the checkpoint
    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"loading {config.model.name} checkpoint from {checkpoint_path}")
        model, msg = load_checkpoint(model, checkpoint_path)
        print("missing keys:")
        print(msg.missing_keys)
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    # Move model to GPUs
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id],find_unused_parameters=True)
        model_without_ddp = model.module

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset ...")  # Note printing only available in the main process
    logger.info(f"Loading dataset from {config.mbeir_data_dir}{config.data_config.train_query_data_path}...")

    #### dataset关于img_preprocess_fn和tokenizer需要与jina v4对上，img_preprocess_fn应该不用管，就lamda (x:x)就行，需要考虑tokenizer在collator的使用
    ### 从模型暴露的接口获取图像预处理函数与分词器，保证与模型要求完全一致（避免数据/模型不一致）。
    # img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    img_preprocess_fn = lambda x : x
    tokenizer = model_without_ddp.get_tokenizer()
    
    ### 获取全局并行度与当前 rank
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    ### 数据集核心构建函数
    train_dataset, trainxxxxx_collector = build_mbeir_dataset_from_config(
        config=config,
        tokenizer=tokenizer,
        img_preprocess_fn=img_preprocess_fn,
        dataset_type=DatasetType.MAIN_TRAIN,
    )
    ##############################调用jinacolla
    #processor = JinaEmbeddingsV4Processor.from_pretrained("/data/jina-v4-local-copy", trust_remote_code=True)
    # train_collector = JinaV4Collator(tokenizer=processor, image_size=(224, 224), mode=Mode.TRAIN)
    train_collector = JinaV4Collator(tokenizer=tokenizer, image_size=(224, 224), mode=Mode.TRAIN)
    ##########################改成jina forward接受的形式（）
    ### DDP 必备：按 rank 均匀切分数据子集并控制 shuffle 同步
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,  # Note: since we use sampler, shuffle should be False
        collate_fn=train_collector,
        drop_last=True,
    )

    enable_eval = config.evaluator.enable_eval
    valid_loader = None
    if enable_eval:
        in_batch_val_dataset, in_batch_val_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.IN_BATCH_VAL,
        )
        in_batch_val_sampler = DistributedSampler(
            dataset=in_batch_val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            sampler=in_batch_val_sampler,
            shuffle=False,  # Note: since we use sampler, shuffle should be False
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )
    else:
        print("In-batch validation is disabled.")

    # Initializing the scheduler
    ### 计算总的 调度步数：
	### •	这里用的是 步数级 退火（而非 epoch 级），即 T_max = 总迭代步数（考虑了梯度累积）。
    ### 可加入 warmup（如 GradualWarmupScheduler 或 get_cosine_schedule_with_warmup），对大模型更稳。
    t_total = (
        len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)

    epoch = 0
    if ckpt_config.resume_training:
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1

    # Training loop
    ### barrier 一次，确保各 rank 均准备好
    # --------- 6-in-1 pre-flight check ---------
    # model_without_ddp.eval()          # 先保证无dropout
    # with torch.no_grad():
    #     # 1. adapter状态
    #     print("=== 1. adapter状态 ===")
    #     print("model.training:", model.training)
    #     print("PeftModel?:", isinstance(model_without_ddp, PeftModel))
    #     if isinstance(model_without_ddp, PeftModel):
    #         cfg = model_without_ddp.peft_config['default']
    #         print("inference_mode:", cfg.inference_mode)

    #     # 2. LoRA权重
    #     print("=== 2. LoRA权重采样 ===")
    #     for n, p in model_without_ddp.named_parameters():
    #         if 'lora' in n and 'retrieval' in n:
    #             print(n, p.mean().item(), p.std().item())
    #             break

    #     # 3. text漂移
    #     print("=== 3. text embedding漂移 ===")
    #     emb_cat = model_without_ddp.encode_text("cat", prompt_name="query",task = "retrieval")
    #     coll  = train_collector
    #     mini_batch = coll(["cat"])          # 伪batch=1
    #     mini_batch = {k: v.to(device) for k,v in mini_batch.items()}
    #     out = model_without_ddp(task_label="retrieval", **mini_batch)
    #     emb_now = out.single_vec_emb[0].cpu().numpy()
    #     cos = float(np.dot(emb_cat, emb_now) / (np.linalg.norm(emb_cat)*np.linalg.norm(emb_now)))
    #     print("cos(cat_pretrained, cat_step0) =", cos)

    #     # 4. 路由
    #     print("=== 4. 路由 ===")
    #     print("active_adapters:", model_without_ddp.active_adapters)
    #     print("task:", model_without_ddp.task)

    # # 5. temperature梯度
    # model_without_ddp.train()
    # model_without_ddp.zero_grad()
    # tmp = model_without_ddp.temperature
    # fake_logits = torch.randn(4,4, device=device) / tmp
    # fake_loss = clip_loss(fake_logits)
    # fake_loss.backward()
    # print("=== 5. temp grad ===", tmp.grad.item())

    # # 6. 单batch loss脚印
    # single = next(iter(train_loader))
    # for k,v in single.items():
    #     if isinstance(v, torch.Tensor): single[k] = v[:4].to(device)
    # logits = model_without_ddp(**single).query_embeddings @ model_without_ddp(**single).target_embeddings.t()
    # logits /= model_without_ddp.temperature
    # print("=== 6. step-0 loss =", clip_loss(logits).item())
    dist.barrier()
    train(
        train_loader,
        valid_loader,
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--uniir_dir",
        type=str,
        default="/data/UniIR",
        help="Path to mbeir directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    ### 使用 OmegaConf 读取 YAML 到 DictConfig
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Initialize distributed training
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    ### utils.init_distributed_mode(args) 内部会解析 RANK/LOCAL_RANK/WORLD_SIZE 等环境变量或启动参数，设置 args.gpu、args.distributed。
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    # Set up wandb
    if config.wandb_config.enabled and utils.is_main_process():
        ### 从 .env 读取 WANDB_API_KEY/PROJECT/ENTITY
        load_dotenv()  # Load .env and get WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY
        wandb_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")

        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found. Ensure it's set in the .env file.")

        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=config.wandb_config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    # Set up logger
    if utils.is_main_process():
        logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
        logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
        if not os.path.exists(logger_out_dir):
            os.makedirs(logger_out_dir, exist_ok=True)
        ### 同时写文件与控制台
        handlers = [logging.FileHandler(logger_out_path), logging.StreamHandler()]
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            level=logging.DEBUG,
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=handlers,
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.info(config)

    main(config)

    # Close wandb
    if config.wandb_config.enabled and utils.is_main_process():
        wandb.finish()

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
