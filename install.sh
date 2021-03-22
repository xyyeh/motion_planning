echo "Installing pytorch"
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

echo "Installing eigen and rbdyn"
# build individually with cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON
# look for the folder that contains setup.py, run ~/anaconda3/envs/pplanner/bin/pip3 install .

echo "Installing Eigen3ToPython"
pushd ./
git clone https://github.com/jrl-umi3218/Eigen3ToPython.git
cd Eigen3ToPython
pip install -r requirements.txt
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON
make -j
cd python3
~/anaconda3/envs/pplanner/bin/pip3 install .
popd
rm -rf Eigen3ToPython

echo "Installing SpaceVecAlg"
pushd ./
git clone --recursive https://github.com/jrl-umi3218/SpaceVecAlg
cd SpaceVecAlg
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON
make -j
cd binding/python/sva/python3
~/anaconda3/envs/pplanner/bin/pip3 install .
popd
rm -rf SpaceVecAlg

echo "Installing RBDyn"
pushd ./
git clone --recursive https://github.com/jrl-umi3218/RBDyn
cd RBDyn
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON
make -j
cd binding/python/rbdyn/python3
~/anaconda3/envs/pplanner/bin/pip3 install .
popd
rm -rf RBDyn