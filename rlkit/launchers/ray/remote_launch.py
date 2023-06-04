import cloudpickle
import uuid
import yaml

import rlkit.launchers.ray_config as config

import ray
from ray.autoscaler.commands import exec_cluster
import rlkit.launchers.ray.local_launch as ray_local_launch

def generate_gcp_config(region=None, head_instance_type=None,
                        worker_instance_type=None, source_image=None,
                        project_id=None, disk_size=100, use_gpu=False,
                        avail_zone=None):
    default_config = config.GCP_CONFIG[use_gpu]
    if region is None:
        region = default_config['REGION']
    if avail_zone is None:
        avail_zone = default_config['REGION_TO_GCP_AVAIL_ZONE'][region]
    if head_instance_type is None:
        head_instance_type = default_config['INSTANCE_TYPE']
    if worker_instance_type is None:
        worker_instance_type = default_config['INSTANCE_TYPE']
    if source_image is None:
        source_image = default_config['SOURCE_IMAGE']
    if project_id is None:
        project_id = default_config['PROJECT_ID']

    guestAccelerators = {}
    if use_gpu:
        guestAccelerators = {
            'guestAccelerators': [
                { 'acceleratorType': 'projects/{project_id}/'
                                    'zones/{zone}/'
                                    'acceleratorTypes/nvidia-tesla-k80'
                                    .format(project_id=project_id, zone=avail_zone),
                'acceleratorCount': 1}
            ]
        }

    return {
        'provider': {
            'type': 'gcp',
            'region': region,
            'project_id': project_id,
            'availability_zone': avail_zone,
        },
        'head_node': {
            'machineType': head_instance_type,
            'disks': [{
                'boot': True,
                'autoDelete': True,
                'type': 'PERSISTENT',
                'initializeParams': {
                    'diskSizeGb': disk_size,
                    'sourceImage': source_image
                }
            }],
            'scheduling': {
                'onHostMaintenance': 'TERMINATE',
            },
            **guestAccelerators.copy()
        },
        'worker_nodes': {
            'machineType': worker_instance_type,
            'disks': [{
                'boot': True,
                'autoDelete': True,
                'type': 'PERSISTENT',
                'initializeParams': {
                    'diskSizeGb': disk_size,
                    'sourceImage': source_image
                }
            }],
            'scheduling': {
                'preemptible': True,
                'onHostMaintenance': 'TERMINATE',
            },
            **guestAccelerators.copy()
        }
    }


def generate_aws_config(region=None, head_instance_type=None,
                        worker_instance_type=None, source_image=None,
                        max_spot_price=None, disk_size=100, avail_zones=None,
                        use_gpu=False):
    default_config = config.AWS_CONFIG[use_gpu]
    if region is None:
        region = default_config['REGION']
    if head_instance_type is None:
        head_instance_type = default_config['INSTANCE_TYPE']
    if worker_instance_type is None:
        worker_instance_type = default_config['INSTANCE_TYPE']
    if source_image is None:
        source_image = default_config['REGION_TO_AWS_IMAGE_ID'][region]
    if max_spot_price is None:
        max_spot_price = default_config['SPOT_PRICE']
    if avail_zones is None:
        avail_zones = default_config['REGION_TO_AWS_AVAIL_ZONE'][region]
    return {
        'provider': {
            'type': 'aws',
            'region': region,
            'availability_zone': avail_zones,
        },
        'head_node': {
            'InstanceType': head_instance_type,
            'ImageId': source_image,
            # 'InstanceMarketOptions': {
                # 'MarketType': 'spot',
                # 'SpotOptions': {
                    # 'MaxPrice': str(max_spot_price),
                # }
            # },
            'BlockDeviceMappings': [
                {
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': disk_size
                    }
                }
            ]
        },
        'worker_nodes': {
            'InstanceType': worker_instance_type,
            'ImageId': source_image,
            'InstanceMarketOptions': {
                'MarketType': 'spot',
                'SpotOptions': {
                    'MaxPrice': str(max_spot_price),
                }
            }
        }
    }


def load_base_config():
    with open("railrl/launchers/ray/base_autoscaler.yaml", "r") as f:
        return yaml.load(f)


def generate_docker_config(docker_image=None, use_gpu=False):
    docker_run_options = []
    # see Ray bug #4403
    docker_run_options.extend([
        "-v /home/ubuntu/ray_results:/home/ubuntu/ray_results",
        "--env LOGNAME=ubuntu",
        "--env HOME=/home/ubuntu",

        # "--shm-size=50gb"
    ])

    python_path = ""
    for mount_info in config.DIR_AND_MOUNT_POINT_MAPPINGS:
        docker_run_options.append(
            '-v {remote_path}:{docker_path}'.format(
                remote_path=mount_info['remote_dir'],
                docker_path=mount_info['mount_point']))
        python_path += mount_info['mount_point'] + ":"
    docker_run_options.append('-e PYTHONPATH={}'.format(python_path))

    if docker_image is None:
        docker_image = config.DOCKER_IMAGE[use_gpu]

    worker_run_options = []
    if use_gpu:
        # Make the worker do the GPU work and let the head node schedule without
        # GPU.
        worker_run_options.append('--runtime=nvidia')
        docker_run_options.append('--runtime=nvidia')
    return {
        'docker': {
            'image': docker_image,
            'container_name': 'ray-docker',
            'run_options': docker_run_options,
            'worker_run_options': worker_run_options,
        }
    }


def launch_remote_experiment(mode, local_launch_variant, use_gpu,
                             remote_launch_variant=None, docker_variant=None,
                             cluster_name=None):
    """
    We generate a ray autoscaler file dynamically here and use `exec_cluster`
    to execute `python /path_to/local_launch.py`, which runs the experiment
    on the instance. The local_launch_variant is pkl'd and uploaded to each node
    """
    if remote_launch_variant is None:
        remote_launch_variant = {}
    if docker_variant is None:
        docker_variant = {}
    remote_launch_variant['use_gpu'] = use_gpu
    docker_variant['use_gpu'] = use_gpu
    """
    Temporary workaround for non-sudo file mounts: specify a remote path on the
    instance machine for each filemount, but when executing inside docker,
    mount with mount_point
    """
    file_mounts = {}
    for mount_pair in config.DIR_AND_MOUNT_POINT_MAPPINGS:
        file_mounts[mount_pair['remote_dir']] = mount_pair['local_dir']

    docker_config = generate_docker_config(**docker_variant)
    provider_specific_config = {
        'gcp': generate_gcp_config,
        'aws': generate_aws_config,
    }[mode](**remote_launch_variant)
    base_config = load_base_config()

    launch_config = {
        **base_config,
        **provider_specific_config,
        **docker_config,
        'file_mounts': file_mounts,
        'initialization_commands': [
            'docker pull {}'.format(docker_config['docker']['image'])
        ],
    }

    # signal to the instance that this should be a remote experiment
    local_launch_variant['from_remote'] = True
    local_launch_variant['resume'] = True
    with open(config.EXPERIMENT_INFO_PKL_FILEPATH, 'wb') as f:
        cloudpickle.dump(local_launch_variant, f)
    with open(config.LAUNCH_FILEPATH, 'w') as f:
        yaml.dump(launch_config, f)
    if cluster_name is None:
        cluster_name = str(uuid.uuid4())
    remote_command = 'python {launch_file}'.format(
        launch_file=ray_local_launch.__file__)
    exec_cluster(config_file=config.LAUNCH_FILEPATH, docker=True,
                 cmd=remote_command, start=True, stop=True, tmux=False,
                 screen=False, override_cluster_name=cluster_name,
                 port_forward=None)
