# fix mmdet3d old version: https://github.com/open-mmlab/mmdetection3d/issues/1332, https://2048.csdn.net/68302f31606a8318e85a36a6.html
# git clone https://github.com/open-mmlab/mmcv.git
# cd mmcv
# git checkout v1.6.1
# MMCV_WITH_OPS=1 pip install -e .
# cd ..
# git clone https://github.com/open-mmlab/mmdetection.git
# cd mmdetection
# git checkout v2.25.1 
# pip install -r requirements/build.txt
# python setup.py develop
# cd ..
git clone  https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
python setup.py develop
cd ..
pip install yapf==0.40.1