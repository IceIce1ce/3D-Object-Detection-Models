# conda install -c fvcore -c iopath -c conda-forge -c pytorch3d -c pytorch fvcore iopath pytorch3d -y
pip install cython opencv-python
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd third_party
git clone https://github.com/facebookresearch/detectron2
python -m pip install -e detectron2
# conda install -c conda-forge scipy seaborn -y
cd UniDepth
pip install -e .
cd ../Grounded-Segment-Anything 
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install scikit-learn
cd ../..
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d 
pip install -e .
cd ..
pip install xformers==0.0.26.post1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install supervision==0.21.0
