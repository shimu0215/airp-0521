# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from types import SimpleNamespace

# ============== 假数据结构（最小实现，模拟 torch_geometric.Data） ==============
class DataObj:
    def __init__(self, pos, batch, atomic_numbers, energy):
        self.pos = pos           # [N,3]
        self.batch = batch       # [N]
        self.atomic_numbers = atomic_numbers
        self.energy = energy     # [num_mols] 这里我们设为 0，表示真最小能量
    def to(self, device):
        self.pos = self.pos.to(device)
        self.batch = self.batch.to(device)
        self.atomic_numbers = self.atomic_numbers.to(device)
        self.energy = self.energy.to(device)
        return self
    def __getitem__(self, key):
        return getattr(self, key)

# ============== 伪造的数据准备（与你项目的 data_prepare 等价接口） ==============
def data_prepare(data, args):
    """
    返回：pos0, _, node_mask, _
    这里简化：node_mask 全 1
    """
    N = data.pos.size(0)
    pos0 = data.pos.clone()
    node_mask = torch.ones(N, 1, dtype=torch.bool, device=args.device)
    return pos0, None, node_mask, None

# ============== 可控“假模型”：predict_PaiNN =========================
def predict_PaiNN(data, node_mask, pretrained_PaiNN, args, pos=None, use_rd=False, ja4=False):
    """
    这里的“模型”能量是 E = 0.5 * sum(||pos||^2)，力 F = -pos。
    - 实际使用中：把这里替换为你的 PaiNN 前向即可。
    - 关键：确保用到传入的 'pos'（更新后的坐标）。
    """
    device = args.device
    mask = node_mask.view(-1).bool()
    if pos is None:
        pos_clean = data.pos[mask].to(device).clone()
    elif use_rd:
        pos_clean = pos.to(device).clone()       # ★ 一定要在 device 上
    else:
        pos_clean = pos[mask].to(device).clone()

    # 假模型能量/力
    E_scalar = 0.5 * (pos_clean**2).sum()             # 标量
    # 把能量做成按分子的张量（与 data.energy 对齐）。这里只有 1 个分子。
    E = E_scalar.unsqueeze(0)                          # [1]
    forces = -pos_clean                                # [N_mask,3]

    pred = {"energy": E, "forces": forces}
    return pred, None

# ============== 能量和梯度封装（保持和你项目类似的接口） ==============
def energy_and_grad(data, node_mask, model, pos_masked, args):
    """
    返回:
      E: [num_mols]（这里是 [1]）
      forces: [N_mask, 3]
    说明：
      * 我们选择“返回 forces（= -∇E）”，更新时做 R <- R + α * forces
        ——这和你现在的写法最贴合，避免符号再反一次。
    """
    pred, _ = predict_PaiNN(data, node_mask, model, args, pos=pos_masked, use_rd=True)
    E = pred["energy"].to(args.device)
    forces = pred["forces"].to(args.device)   # forces = -grad
    return E, forces

# ============== 纯梯度下降 + 回溯线搜索（单调下降的快速验证） ===========
@torch.no_grad()
def optimize_gd_with_backtracking(model, data, args,
                                  max_steps=50, alpha_init=1e-1,
                                  shrink=0.5, max_backtrack=10):
    device = args.device
    data = data.to(device)

    pos0, _, node_mask, _ = data_prepare(data, args)
    mask = node_mask.view(-1).bool()

    # 起点：随便挑个偏离零点的坐标（真实最小在 0）
    R = data.pos[mask].detach().clone()       # [N_mask, 3]

    # 初始能量
    E, forces = energy_and_grad(data, node_mask, model, R, args)
    print(f"Step 0: E = {E.item():.6e}")

    for step in range(1, max_steps + 1):
        alpha = torch.full_like(R[:, :1], alpha_init, device=device)
        direction = forces                   # forces = -∇E（下降方向）

        # 回溯线搜索：保证 E_new <= E
        ok = False
        for _ in range(max_backtrack):
            R_try = (R + alpha * direction).detach()
            E_try, _ = energy_and_grad(data, node_mask, model, R_try, args)
            if (E_try <= E).all():
                ok = True
                break
            alpha = alpha * shrink           # 步长减半重试

        if not ok:
            print(f"  step {step}: backtracking failed, stop.")
            break

        # 接受步 & 更新
        R = R_try
        E, forces = energy_and_grad(data, node_mask, model, R, args)
        print(f"Step {step}: E = {E.item():.6e}")

    return R, E

# ================================ 主程序 ================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # 设备/参数
    args = SimpleNamespace(device="cpu")  # 改成 "cuda" 也可以
    N = 12                                # 原子数
    num_mols = 1                          # 这里用 1 个分子

    # 构造一个初始的“非零”坐标（随机），真最小能量对应 pos = 0
    init_pos = torch.randn(N, 3) * 0.5 + 1.0  # 偏离零点
    batch = torch.zeros(N, dtype=torch.long)  # 全部属于同一分子
    Z = torch.full((N,), 6)                   # 原子序数（碳），这里只是占位
    E_label = torch.tensor([0.0])             # 真最小能量设为 0（仅用于打印/对齐）

    data = DataObj(pos=init_pos, batch=batch, atomic_numbers=Z, energy=E_label)

    # “模型”占位（这例子里没用到）
    model = None

    print("=== Quick sanity check: gradient descent with backtracking ===")
    R_final, E_final = optimize_gd_with_backtracking(
        model, data, args, max_steps=20, alpha_init=0.5, shrink=0.5, max_backtrack=10
    )
    print(f"\nDone. Final E = {E_final.item():.6e}")