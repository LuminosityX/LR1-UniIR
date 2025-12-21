# debug_jina_v4.py
import torch, transformers
from rich.console import Console
from rich.table import Table
from typing import Dict, Any

console = Console()

# ---------- 1. 在 collator 出口打印 ----------
def wrap_collator(collator_class):
    orig_call = collator_class.__call__
    def _debug_collator(self, batch):
        out = orig_call(self, batch)
        console.rule("[bold cyan]Collator 出口", style="cyan")
        _print_batch(out, "collator")
        return out
    collator_class.__call__ = _debug_collator
    return collator_class

# ---------- 2. 在 forward 入口打印 ----------
def wrap_forward(forward_func):
    def _debug_forward(self, **batch):
        console.rule("[bold magenta]Forward 入口", style="magenta")
        _print_batch(batch, "forward")
        return forward_func(self, **batch)
    return _debug_forward

# ---------- 3. 统一打印格式 ----------
def _print_batch(batch: Dict[str, Any], stage: str):
    table = Table(title=f"{stage} batch 摘要", show_header=True, header_style="bold white")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Shape / Type", style="green")
    table.add_column("Modality 样例", style="yellow")

    B = len(batch.get("query_type", []))
    for k, v in batch.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            table.add_row(k, str(list(v.shape)), "")
        elif isinstance(v, (transformers.BatchEncoding, dict)) and "input_ids" in v:
            table.add_row(k, str(list(v["input_ids"].shape)), "")
        elif k.endswith("_type"):          # modality 标签
            table.add_row(k, f"List[str] len={B}", str(v[:3]))
        elif k in ("query_pixel_values", "target_pixel_values"):
            imgs = [type(p).__name__ if p else "None" for p in v]
            table.add_row(k, f"List[PIL|None] len={B}", str(imgs[:3]))
        else:
            table.add_row(k, str(type(v).__name__), "")
    console.print(table)

    # ---------- 4. 顺序一致性检查 ----------
    if stage == "forward" and "query_type" in batch:
        q_types = batch["query_type"]
        t_types = batch["target_type"]
        q_ids   = batch["query_input_ids"]
        console.print(f"[bold blue]样本顺序快速检查[/]")
        for i in range(min(3, len(q_types))):
            q_txt = self.tokenizer.decode(q_ids[i], skip_special_tokens=True)[:40]
            console.print(f"  {i} | q_type={q_types[i]} | q_txt={q_txt}…")

# ---------- 5. 应用 ----------
def apply_debug(model_class, collator_class):
    wrap_collator(collator_class)
    model_class.forward = wrap_forward(model_class.forward)