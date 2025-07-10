pip install git+https://github.com/facebookresearch/segment-anything.git
git clone https://github.com/lpiccinelli-eth/UniDepth
cd UniDepth
pip install -e .
cd ..
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
