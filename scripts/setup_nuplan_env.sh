#!/bin/bash
# Build a CPU-only Python 3.9 environment for the nuPlan simulation stack.
#
# Why a second environment at all: nuplan-devkit requires Python >= 3.9 and the project's `edge`
# environment is 3.8.20, tied to NVIDIA's Jetson torch build.  Track B does not need torch --
# nuplan-devkit keeps its torch dependencies in a separate requirements_torch.txt, and both
# PDM-Closed and IDMPlanner are rule-based -- so the simulation stack installs without it and
# the Jetson torch build is left untouched.
#
# Native dependencies (GDAL via Fiona/rasterio/pyogrio, Shapely, rtree, pyarrow, casadi) come
# from conda-forge, which publishes aarch64 builds; only pure-Python packages come from pip.
# Version pins the devkit requests but conda cannot satisfy on aarch64 are recorded as
# deviations in third_party/PROVENANCE.md rather than forced.
set -e
ENV=nuplan
MAMBA=$HOME/miniforge3/bin/mamba

$MAMBA create -y -n $ENV -c conda-forge python=3.9 \
  numpy=1.23 scipy pandas pyarrow "shapely>=2.0" geopandas fiona rasterio pyogrio rtree \
  py-opencv casadi scikit-learn matplotlib pillow tqdm joblib cachetools psutil requests \
  urllib3 sympy typer ujson aiofiles nest-asyncio sqlalchemy=1.4.27 bokeh=2.4.3 \
  "hydra-core>=1.1" pyquaternion tornado jupyter

P=$HOME/miniforge3/envs/$ENV/bin/python
$P -m pip install --no-input retry positional-encodings control guppy3 pyinstrument \
  aioboto3 boto3 s3fs moto mock docker grpcio grpcio-tools || true

# the two checkouts, installed without letting pip re-resolve the pinned dependency graph
$P -m pip install --no-deps -e third_party/nuplan_devkit
$P -m pip install --no-deps -e third_party/tuplan_garage
echo "NUPLAN ENV BUILT"
$P -c "import nuplan, tuplan_garage; print('import ok:', nuplan.__file__)"
