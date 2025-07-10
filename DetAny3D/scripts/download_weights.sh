mkdir -p checkpoints
cd checkpoints
mkdir -p sam_ckpts
cd sam_ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
mkdir -p unidepth_ckpts
mkdir -p dino_ckpts
mkdir -p detany3d_ckpts
cd ..
