import torch
from torch.cuda.amp import autocast
import transformers
from PIL import Image
from torch import nn
from models.jina_v4t import utils
import torch.nn.functional as F
import argparse
import logging
import os
import random
import time
import datetime
from transformers import AutoModel
# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0
######原细节不动，改调用和batch搬运
# def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler,
#                     global_step, scaler, config):

#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     header = "Train Epoch: [{}]".format(epoch)
#     print_freq = config.trainer_config.print_freq

#     accumulation_steps = config.trainer_config.gradient_accumulation_steps
#     accumulation_counter = 0

#     # -----------  新增：400 step checkpoint  ----------
#     # 注意：这里 global_step 是优化步（参数更新一次算一步）
#     # 如果你希望 400 个“micro-step”就保存，把条件改成
#     # (global_step * accumulation_steps + accumulation_counter) == 400
#     # -------------------------------------------------
#     for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         ...  # 原封不动保留你原来的搬运数据、alpha、autocast、loss、acc 代码

#         # 梯度累加部分也不动
#         accumulation_counter += 1
#         if accumulation_counter == accumulation_steps:
#             global_step += 1
#             scaler.step(optimizer)
#             scaler.update()
#             model.zero_grad()
#             scheduler.step()
#             accumulation_counter = 0

#             # ======  关键：400 优化步时保存  ======
#             if global_step == 400 and utils.is_main_process():
#                 save_checkpoint(model, optimizer, scheduler, epoch, scaler,
#                                 config, suffix="step400")
#                 print(">>> 400-step checkpoint saved.")
#                 # 如果只想跑到 400 step 就结束，把下面 break 放开
#                 # break
#             # =====================================

#         metric_logger.update(loss=loss.item() * accumulation_steps)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

#     metric_logger.synchronize_between_processes()
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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

def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    model.train()

    ### 构建度量记录器，定义要跟踪的指标：学习率、loss、in-batch accuracy；window_size=1 表示不做滑窗平滑。
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0
    ### 迭代 dataloader，并由 log_every 控制打印频率。
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key, value in batch.items():
            #### 换成jina v4后需要考虑这里的修改
            ### 将 batch 中的张量或 HF BatchEncoding 内部张量搬到当前 GPU。
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(gpu_id, non_blocking=True)
            # elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
            #     for k, v in value.items():
            #         batch[key][k] = v.to(gpu_id)
            elif isinstance(value, (transformers.BatchEncoding, dict)):
                for k, v in value.items():
                    if torch.is_tensor(v):
                        batch[key][k] = v.to(gpu_id, non_blocking=True)
        # if i % 10 == 0 and utils.is_main_process():
        #     print('batch row idx', list(range(len(batch['query_input_ids'][:4]))))
        #     print('query feat idx', batch['query_input_ids'][:, 0].tolist())   # 随便拿一个 token 当指纹
        #     print('target feat idx', batch['target_input_ids'][:, 0].tolist())
        ### alpha 线性 warmup（仅第 0 个 epoch）：从 0 线性涨到设定 alpha，之后各 epoch 固定为 alpha。
        ### alpha 控制队列对比损失的权重。
        if epoch > 0:
            alpha = config.model.alpha
        else:
            alpha = config.model.alpha * min(1, i / len(data_loader))

        # autocast for mixed precision
        with autocast():
            # outputs = model(batch=batch, alpha=alpha)
            # loss = outputs["loss"]
            # inbatch_accuracy = outputs["accuracy"]
            # output = model(**batch, alpha=alpha)
            
            # 改成
            batch["alpha"] = alpha          # 塞进字典
            #print("[batch-forward]",batch)
            output = model(**batch)         # 不再显式传 alpha
            loss = output.loss
            

            # # ---------- 对比学习“卡住”诊断 ----------
            # if i % 200 == 0:                       # 每 200 步诊断一次，可调
            #     with torch.no_grad():
            #         f_q   = output.query_embeddings          # [B, D]
            #         f_t   = output.target_embeddings         # [B, D]
            #         # 1) 特征模长
            #         f_norm_q = f_q.norm(dim=1).mean().item()
            #         f_norm_t = f_t.norm(dim=1).mean().item()
            #         # 2) 特征 cosine 矩阵
            #         cos = torch.matmul(F.normalize(f_q, dim=1),
            #                         F.normalize(f_t, dim=1).t())  # [B,B]
            #         cos_mean = cos.mean().item()
            #         cos_off  = (cos - torch.eye(cos.shape[0], device=cos.device)).abs().mean().item()
            #         # 3) logits 范围 & τ（如果 loss 里有 τ）
            #         logits = f_q @ f_t.t() / model.module.temperature  ###############
            #         print(f'[diag] T={model.module.temperature.item():.3f}  f_norm=({f_norm_q:.3f},{f_norm_t:.3f}) '
            #             f'cos={cos_mean:.3f} off-diag={cos_off:.3f} '
            #             f'logits=({logits.min().item():.3f},{logits.max().item():.3f})')
            #         print(f'[diag] queue_ptr={model.module.queue_ptr.item()} '
            #             f'queue_q_std={model.module.queue_q.std().item():.4f}')
            # ---------- 对比学习“卡住”诊断（队列版） ----------
            # ---------- 对比学习诊断（与队列 loss 同口径） ----------
            if i % 50 == 0:
                with torch.no_grad():
                    B = output.query_embeddings.size(0)
                    f_q = F.normalize(output.query_embeddings, dim=1)   # 与训练一致
                    f_t = F.normalize(output.target_embeddings, dim=1)

                    # 1. 特征模长（应≈1）
                    f_norm_q = f_q.norm(dim=1).mean().item()
                    f_norm_t = f_t.norm(dim=1).mean().item()

                    # 2. in-batch cosine
                    cos_inbatch = torch.matmul(f_q, f_t.t())            # (B,B)
                    cos_mean = cos_inbatch.mean().item()
                    cos_off  = (cos_inbatch - torch.eye(B, device=cos_inbatch.device)).abs().mean().item()

                    # 3. 队列负例相似度（与 loss 同一批）
                    q_neg = model.module.queue_q.clone()                # (K,D)
                    t_neg = model.module.queue_t.clone()
                    queue_cos_q = torch.matmul(f_q, t_neg.T).mean().item()   # q vs 队列
                    queue_cos_t = torch.matmul(f_t, q_neg.T).mean().item()   # t vs 队列

                    # 4. 有效 logits（与训练完全一致）
                    all_t = torch.cat([f_t, t_neg], dim=0)              # (B+K, D)
                    all_q = torch.cat([f_q, q_neg], dim=0)              # (B+K, D)
                    logits_q = torch.matmul(f_q, all_t.T) / model.module.temperature   # (B, B+K)
                    logits_t = torch.matmul(f_t, all_q.T) / model.module.temperature
                    lmin = min(logits_q.min().item(), logits_t.min().item())
                    lmax = max(logits_q.max().item(), logits_t.max().item())

                    # 5. 正例 / 负例均值
                    pos_sim  = cos_inbatch.diag().mean().item()
                    neg_in   = (cos_inbatch.sum() - cos_inbatch.diag().sum()) / (B*(B-1))
                    queue_neg_sim = (queue_cos_q + queue_cos_t) / 2.

                    print(f'[diag] T={model.module.temperature.item():.3f} '
                        f'f_norm=({f_norm_q:.3f},{f_norm_t:.3f}) '
                        f'cos_in={cos_mean:.3f} off={cos_off:.3f} '
                        f'pos={pos_sim:.3f} neg_in={neg_in:.3f} neg_q={queue_neg_sim:.3f} '
                        f'logits=({lmin:.2f},{lmax:.2f})')
                    print(f'[diag] queue_ptr={model.module.queue_ptr.item()} '
                        f'queue_q_std={model.module.queue_q.std().item():.4f}')
            # if i % 50 == 0:
            #     with torch.no_grad():
            #         B = output.query_embeddings.size(0)
            #         f_q = output.query_embeddings   # 已归一化
            #         f_t = output.target_embeddings

            #         # 1. 特征模长（归一化后应≈1）
            #         f_norm_q = f_q.norm(dim=1).mean().item()
            #         f_norm_t = f_t.norm(dim=1).mean().item()

            #         # 2. in-batch  cosine 矩阵
            #         cos_inbatch = torch.matmul(f_q, f_t.t())            # (B,B)
            #         cos_mean = cos_inbatch.mean().item()
            #         cos_off  = (cos_inbatch - torch.eye(B, device=cos_inbatch.device)).abs().mean().item()

            #         # 3. 队列负例相似度（抽样前 100 个做均值，避免 K 太大）
            #         q_neg = model.module.queue_q.clone()                # (D,K)
            #         t_neg = model.module.queue_t.clone()
            #         with torch.no_grad():
            #             queue_cos_q = torch.matmul(f_q, q_neg).mean().item()   # q  vs 队列负例
            #             queue_cos_t = torch.matmul(f_t, t_neg).mean().item()   # t  vs 队列负例

            #         # 4. 有效 logits 范围（in-batch + queue）
            #         logits_q = torch.cat([cos_inbatch, torch.matmul(f_q, t_neg)], dim=1) / model.module.temperature
            #         logits_t = torch.cat([cos_inbatch.t(), torch.matmul(f_t, q_neg)], dim=1) / model.module.temperature
            #         lmin = min(logits_q.min().item(), logits_t.min().item())
            #         lmax = max(logits_q.max().item(), logits_t.max().item())

            #         # 5. 正例/负例均值对比
            #         pos_sim  = cos_inbatch.diag().mean().item()         # 正例
            #         neg_sim  = (cos_inbatch.sum() - cos_inbatch.diag().sum()) / (B*(B-1))  # in-batch 负例
            #         queue_neg_sim = (queue_cos_q + queue_cos_t) / 2.    # 队列负例

            #         print(f'[diag] T={model.module.temperature.item():.3f} '
            #             f'f_norm=({f_norm_q:.3f},{f_norm_t:.3f}) '
            #             f'cos_in={cos_mean:.3f} off={cos_off:.3f} '
            #             f'pos={pos_sim:.3f} neg_in={neg_sim:.3f} neg_q={queue_neg_sim:.3f} '
            #             f'logits=({lmin:.2f},{lmax:.2f})')
            #         print(f'[diag] queue_ptr={model.module.queue_ptr.item()} '
            #             f'queue_q_std={model.module.queue_q.std().item():.4f}')
            with torch.no_grad():
                q = output.query_embeddings
                t = output.target_embeddings
                logits = q @ t.t()          # [B, B]
                labels = torch.arange(len(logits), device=logits.device)
                acc = (logits.argmax(dim=1) == labels).float().mean()
                inbatch_accuracy = acc

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        ### 在 backward 前缩放 loss：累积 accumulation_steps 次后，整体有效梯度与不缩放一次 step 等价。
        loss = loss / accumulation_steps

        # Use scaler for backward
        ### AMP 的缩放反传：避免半精下梯度下溢。
        scaler.scale(loss).backward()
        # for n, p in model.named_parameters():
        #     if "lora" in n.lower() and p.requires_grad:
        #         print(n, "[debug]grad norm:", None if p.grad is None else f"{p.grad.norm().item():.6f}")

        ### 当累积次数达到阈值：
        # 	•	步进计数（global_step 这里仅内部累加，外部未使用）。
        # 	•	scaler.step(optimizer)：尝试更新参数；若出现 inf/nan，step 会跳过。
        # 	•	scaler.update()：调整缩放系数。
        # 	•	model.zero_grad()：清空梯度（✅ 建议改为 optimizer.zero_grad(set_to_none=True)，更省内存、快）。
        # 	•	scheduler.step()：每个优化步后调度学习率。
        #       请确保 CosineAnnealingLR(T_max) 的 T_max 按优化步计算（主程序已按迭代步计算，合理）。
        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            model.zero_grad()
            scheduler.step()
            accumulation_counter = 0
            with torch.no_grad():
                # model.module.temperature.clamp_(min=0.01)
                model.module.temperature.clamp_(0.01, 0.5) 
            # print("[global_step]",global_step)
            # with torch.no_grad():
            #     for n,p in model.named_parameters():
            #         if 'lora_B' in n and 'retrieval' in n:
            #             dead = (p.abs() < 1e-7).sum().item()
            #             print(f"STEP-{global_step}  {n}  零元素={dead}/{p.numel()}  mean={p.mean():.2e}  grad_mean={p.grad.abs().mean():.2e}")
            #             break
            if global_step == 100 :
                print("save_checkpoint 前模型类型:", type(model))
                real_model = model.module
                
                # --------- 新增诊断 ---------
                print("Peft type:", type(real_model))
                print("Peft config keys:", list(real_model.peft_config.keys()) if hasattr(real_model, 'peft_config') else 'None')
                # 随便抽一个 LoRA 参数看是否存在
                sample_key = [k for k in real_model.state_dict().keys() if 'lora_B' in k]
                print("Sample LoRA key:", sample_key[:1])
                print("save_checkpoint 前模型类型:", type(real_model))  # 应该是 PeftModel 或其子类
                with torch.no_grad():
                    sample_key = [k for k in real_model.state_dict() if "lora_B.default.retrieval.weight" in k][0]
                    print("保存前瞬间 lora_B max:", real_model.state_dict()[sample_key].max().item())
                # 2. 传入真实模型
                save_checkpoint(real_model, optimizer, scheduler, epoch, scaler,
                                config, suffix="stepqueuelosslast")
                # 你原来的 save_checkpoint 函数支持 suffix 参数
                # save_checkpoint(model, optimizer, scheduler, epoch, scaler,
                #                 config, suffix="stepqueuelossdecay3")
                
                print(">>> 4-step checkpoint saved.")

        metric_logger.update(loss=loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # TODO: might need to loop through all param groups
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    ### 多进程聚合指标（例如取均值），打印并返回每个 meter 的全局平均。
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

### 评估（不反传）
# @torch.no_grad()
# def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
#     header = "Test:"
#     print_freq = config.evaluator.print_freq

#     # Save model states
#     saved_state = model_without_ddp.state_dict()

#     # Clear model queue states
#     ### 关键：清理对比学习队列（query/cand/idx 队列与指针），避免训练阶段的队列状态污染评估。
#     model_without_ddp.query_queue.copy_(torch.randn_like(model_without_ddp.query_queue))
#     model_without_ddp.cand_queue.copy_(torch.randn_like(model_without_ddp.cand_queue))
#     model_without_ddp.idx_queue.copy_(torch.full_like(model_without_ddp.idx_queue, -100))
#     model_without_ddp.new_ptr_queue.zero_()
#     print("Cleared model queue states.")

#     ### 与训练同样，把 batch 搬到 GPU。
#     for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         for key, value in batch.items():
#             if isinstance(value, torch.Tensor):
#                 batch[key] = value.to(gpu_id, non_blocking=True)
#             elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
#                 for k, v in value.items():
#                     batch[key][k] = v.to(gpu_id)

#         alpha = config.model.alpha * min(1, i / len(data_loader))

#         # autocast for mixed precision
#         with autocast():
#             outputs = model(batch=batch, alpha=alpha)
#             loss = outputs["loss"]
#             inbatch_accuracy = outputs["accuracy"]

#         metric_logger.update(loss=loss.item())
#         metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger.global_avg())

#     # Restore model states from the saved variables
#     model_without_ddp.load_state_dict(saved_state)
#     print("Restored model queue states and model states from the saved variables.")

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    # （可选）保存状态，以便评估后恢复
    saved_state = model_without_ddp.state_dict()

    # ******* 删除 BLIP 队列清理（Jina-V4 无队列）*******
    # model_without_ddp.query_queue.copy_(...)
    # model_without_ddp.cand_queue.copy_(...)
    # model_without_ddp.idx_queue.copy_(...)
    # model_without_ddp.new_ptr_queue.zero_()
    # print("Cleared model queue states.")

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ---- 搬运到 GPU ----
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(gpu_id, non_blocking=True)
            elif isinstance(value, (transformers.BatchEncoding, dict)):
                for k, v in value.items():
                    if torch.is_tensor(v):
                        batch[key][k] = v.to(gpu_id, non_blocking=True)

        alpha = config.model.alpha * min(1, i / len(data_loader))

        # ---- 前向 ----
        with autocast():
            output = model(**batch, alpha=alpha, return_loss=True)
            loss = output.loss
            # in-batch top-1 accuracy
            logits = output.query_embeddings @ output.target_embeddings.t()
            labels = torch.arange(len(logits), device=logits.device)
            inbatch_accuracy = (logits.argmax(dim=1) == labels).float().mean()

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # ---- 多进程聚合 ----
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    # 恢复状态（可选，无队列也可保留）
    model_without_ddp.load_state_dict(saved_state)
    print("Restored model states.")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
