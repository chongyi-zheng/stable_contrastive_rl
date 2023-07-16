import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from rlkit.visualization import plot_util as plot

plt.style.use("ggplot")
import matplotlib

plot.configure_matplotlib(matplotlib)


def main(args):
    exp_dirs = [os.path.expanduser(exp_dir) for exp_dir in args.exp_dirs]
    exps = plot.load_exps(exp_dirs, suppress_output=True)

    print("Number of experiments: {}".format(len(exps)))

    filtered_exps = plot.filter_exps(exps, plot.filter_by_flat_params({
        #     'network_type': 'vqvae',
        #     'reward_kwargs.epsilon': 3.0,
        #     'trainer_kwargs.beta': 0.01,
        #     'trainer_kwargs.quantile': 0.7,
        #     'trainer_kwargs.bc': False,
        #     'source_target_split': True,
        #     'sample_source_fraction': 0.3,
        #     'trainer_kwargs.quantile': 0.7,
        #     'trainer_kwargs.kld_weight': 0.1,
        #     'qf_kwargs.dropout_rate': None,
        #     'expl_planner_kwargs.values_weight': 0.001,
        #     'eval_seeds': 60,
        #     'method_name': 'modelfree',
        #     'expl_planner_type': 'hierarchical',

        #     'expl_planner_kwargs.prior_weight': 0.01,
        #     'expl_planner_kwargs.values_weight': 0.001,
    }))
    print("Number of filtered experiments: {}".format(len(filtered_exps)))

    # vary = [
    #     #     'trainer_kwargs.beta',
    #     #     'reward_kwargs.epsilon',
    #     #     'trainer_kwargs.quantile',
    #     #     'trainer_kwargs.kld_weight',
    #     #     'trainer_kwargs.bc',
    #     #     'trainer_kwargs.plan_vf_weight',
    #     #     'exp_name',
    #     #     'trainer_kwargs.reward_transform_kwargs.b',
    #
    #     #     'sample_source_fraction',
    #     #     'trainer_kwargs.affordance_beta',
    #     #     'replay_buffer_kwargs.max_future_dt',
    #     #     'replay_buffer_kwargs.min_future_dt',
    #     #     'trial_name',
    #
    #     #     'qf_kwargs.dropout_rate',
    #     #     'qf_kwargs.layer_norm',
    #
    #     #     'algo_kwargs.num_online_trains_per_train_loop',
    #     #     'replay_buffer_kwargs.fraction_future_context',
    #     #     'online_offline_split_replay_buffer_kwargs.offline_replay_buffer_kwargs.fraction_future_context',
    #     #     'online_offline_split_replay_buffer_kwargs.online_replay_buffer_kwargs.fraction_future_context',
    #     #     'expl_planner_kwargs.prior_weight',
    #     #     'expl_planner_kwargs.values_weight',
    #
    #     #     'use_expl_planner',
    #     'method_name',
    #     'run_id',
    #     # 'trainer_kwargs.bc_coef',
    #     # 'exp_name',
    # ]
    vary = args.vary

    split = [
        #         'expl_planner_type',
        #     'eval_seeds',s
        #     'trainer_kwargs.beta',
        #     'reward_kwargs.epsilon',
        #     'trainer_kwargs.quantile',
        #     'trainer_kwargs.kld_weight',
        #     'exp_name',

        #     'sample_source_fraction',
        # 'eval_seeds'
        #     'expl_planner_kwargs.prior_weight',
        #     'expl_planner_kwargs.values_weight',
    ]
    default_vary = {}

    plot.split(filtered_exps,
               args.stats,
               vary,
               split,
               default_vary=default_vary,
               smooth=plot.padded_ma_filter(10),
               figsize=args.fig_size,
               num_x_plots=4,
               print_max=False,
               #             xlim=(-100, 150),
               #            ylim=(0.0, 1.0),
               # ylim=(-400, 0),
               #            ylim=(-.2, .2),
               #            method_order=(0, 2, 1),
               symlog_scale_keys=args.symlog_scale_stats,
               log_scale_keys=args.log_scale_stats,
               zoom_in_keys=args.zoom_in_stats,
               zoom_in_x_lims=args.zoom_in_x_lims,
               zoom_in_y_lims=args.zoom_in_y_lims,
               plot_seeds=args.plot_seeds,
               plot_error_bars=args.plot_error_bars,
               print_final=False,
               print_min=False,
               print_plot=True,
               xlabel='epoch',
               )

    plt.tight_layout()

    # plt.savefig("/data/chongyiz/offline_c_learning/railrl-private/logs/figures/iql_vib.png")
    fig_dir = os.path.abspath(os.path.expanduser(args.fig_dir))
    os.makedirs(fig_dir, exist_ok=True)
    fig_filepath = os.path.join(fig_dir, args.fig_filename)
    plt.savefig(fig_filepath + '.png')
    print("Save figure to: {}".format(fig_filepath + '.png'))


if __name__ == "__main__":
    def float_pair(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        splited_float = tuple(map(float, splited_s))
        return splited_float[0], splited_float[1]
    def int_pair(s):
        splited_s = s.split(',')
        assert splited_s, 'invalid string pair'
        splited_float = tuple(map(int, splited_s))
        return splited_float[0], splited_float[1]
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", type=str, nargs="+",
                        default=["~/offline_c_learning/railrl_logs/contrastive_nce_vib/pretrained/run0/id0"])
    parser.add_argument('--stats', type=str, nargs='+', default=[
        'eval/state_desired_goal/final/overall_success Mean',
        'expl/state_desired_goal/final/overall_success Mean',
        'trainer/train/Alpha Loss',
        'trainer/train/Policy Loss',
        'trainer/train/QF1 Loss',
        'trainer/train/QF2 Loss',
        'trainer/train/VIB Loss',
        'trainer/train/Behavioral Cloning Loss',
        'trainer/train/Policy Loss/Actor Q Loss',
        'trainer/train/Policy Loss/GCBC Loss',
    ])
    parser.add_argument('--vary', type=str, nargs='+', default=[
        'method_name',
        # 'run_id',
    ])
    parser.add_argument('--plot_seeds', type=str2bool, default=True)
    parser.add_argument('--plot_error_bars', type=str2bool, default=False)
    parser.add_argument('--fig_size', type=int_pair, default=(12, 8))
    parser.add_argument("--symlog_scale_stats", type=str, nargs="+", default=[])
    parser.add_argument("--log_scale_stats", type=str, nargs="+", default=[])
    parser.add_argument("--zoom_in_stats", type=str, nargs="+", default=[])
    parser.add_argument("--zoom_in_x_lims", type=float_pair, nargs="+", default=[])
    parser.add_argument("--zoom_in_y_lims", type=float_pair, nargs="+", default=[])
    parser.add_argument("--fig_dir", type=str, default="~/offline_c_learning/railrl-private/results/contrastive_rl")
    parser.add_argument("--fig_filename", type=str, default="contrastive_rl")
    args = parser.parse_args()

    main(args)
