import torch, argparse
from ppo_er_unified import Agent, UNIFIED_OBS_DIM, evaluate_task

SEQ = ["PegInsertionSide-v1", "StackCube-v1", "PullCube-v1", "PickCube-v1",
       "LiftPegUpright-v1", "PushCube-v1", "PokeCube-v1", "PullCubeTool-v1"]
TID = SEQ.index("PickCube-v1")

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--eps-list", default="20,100")
ap.add_argument("--private-base", type=int, default=0)
a = ap.parse_args()

sd = torch.load(a.ckpt, map_location="cpu")
agent = Agent(UNIFIED_OBS_DIM, 8, n_heads=len(SEQ), private_base_ids=(a.private_base,))
missing, unexpected = agent.load_state_dict(sd, strict=False)
print(f"{a.ckpt}\n  missing={list(missing)} unexpected={list(unexpected)}")
agent = agent.cuda().eval()

for n in [int(x) for x in a.eps_list.split(",")]:
    sr, r = evaluate_task(agent, "PickCube-v1", n, backend="gpu", device="cuda", tid=TID)
    print(f"  PickCube (tid={TID}) eps={n:3d} SR={sr:5.1f}% avg_R={r:6.2f}")
