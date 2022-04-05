import os

wv_dir = r"./pretrained_wvs"

cur_dir = "./"

snips_fdir = "SnipsCoach"

save_dir = os.path.join(cur_dir, "logs")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model_dir = os.path.join(cur_dir, "saves")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
