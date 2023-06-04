# Change these things
CODE_DIRS_TO_MOUNT = [
    '/home/user/python/module/one',
    '/home/user/python/module/two',
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/user/.mujoco/',
        mount_point='/root/.mujoco',
    ),
]
LOCAL_LOG_DIR = '/home/user/git/path/to/save/data/'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/user/path/to/railrl/scripts/run_experiment_from_doodad.py'
)
# If not set, default will be chosen by doodad
# AWS_S3_PATH = 's3://bucket/directory

AWS_S3_PATH = 's3://2-12-2017.railrl.vitchyr.rail.bucket/doodad/logs-12-01-2017'

# You probably don't need to change things below
# Specifically, the docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-vitchyr'
INSTANCE_TYPE = 'c4.large'
SPOT_PRICE = 0.03

GPU_DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-vitchyr-gpu'
GPU_INSTANCE_TYPE = 'g2.2xlarge'
GPU_SPOT_PRICE = 0.5
# These AMI images have the docker images already installed.
REGION_TO_GPU_AWS_IMAGE_ID = {
    'us-west-1': "ami-874378e7",
    'us-east-1': "ami-0ef1b374",
}

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'


"""
Slurm Settings
"""
SINGULARITY_IMAGE = '/home/vitchyr/singularity/railrl-vitchyr-v2.img'
SINGULARITY_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin'
]
SLURM_CPU_CONFIG = dict(
    account_name='fc_rail',
    partition='savio',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)
SLURM_GPU_CONFIG = dict(
    account_name='fc_rail',
    partition='savio2_1080ti',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)


"""
Slurm Script Settings

These are basically the same settings as above, but for the remote machine
where you will be running the generated script.
"""
SSS_CODE_DIRS_TO_MOUNT = [
    '/global/home/users/vitchyr/git/railrl',
    '/global/home/users/vitchyr/git/multiworld',
    '/global/home/users/vitchyr/git/doodad',
]
SSS_DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/global/home/users/vitchyr/.mujoco',
        mount_point='/root/.mujoco',
    ),
]
SSS_LOG_DIR = '/global/scratch/vitchyr/doodad-log'

SSS_IMAGE = '/global/scratch/vitchyr/singularity_imgs/railrl-vitchyr-v2.img'
SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/global/home/users/vitchyr/git/railrl/scripts/run_experiment_from_doodad.py'
)
SSS_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/vitchyr/.mujoco/mjpro150/bin'
]

"""
GCP Settings
"""
GCP_IMAGE_NAME = 'railrl-torch-4-cpu'
GCP_GPU_IMAGE_NAME = 'railrl-torch4cuda9'
GCP_BUCKET_NAME = 'railrl-steven'

GCP_DEFAULT_KWARGS = dict(
    zone='us-west2-c',
    instance_type='n1-standard-4',
    image_project='railrl-private-gcp',
    terminate=True,
    preemptible=True,
    gpu_kwargs=dict(
        gpu_model='nvidia-tesla-p4',
        num_gpu=1,
    )
)
