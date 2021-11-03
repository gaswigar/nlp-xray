import argparse
import os
import json
import sys

from . import model

def _parse_arguments(argv):
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        help='which model type to use',
        type=str, default='mobilenetv2')
    parser.add_argument(
        '--epoch',
        help='number of epochs to use',
        type=int, default=5)
    parser.add_argument(
        '--steps_per_epoch',
        help='number of steps per epoch to use',
        type=int, default=10)
    parser.add_argument(
        '--job_dir',
        help='directory where to save the model',
        type=str, default='xray_models/')
    parser.add_argument(
        '--nrows',
        help='number of total rows desired accross test, validation, and test set',
        type=int, default=None)
    parser.add_argument('--loss_class_weighted', dest='loss_class_weighted', action='store_true')
    parser.add_argument('--loss_not_class_weighted', dest='loss_class_weighted', action='store_false')
    parser.set_defaults(loss_class_weighted=True)
    return parser.parse_known_args(argv)

def main():
    """
    This function will parse command line arguments and kick model training
    """
    args = _parse_arguments(sys.argv[1:])[0]
    print(f'Training with Arguments:{args}')
    trial_id = json.loads(
        os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '')
    output_path = args.job_dir if not trial_id else args.job_dir+'/'
    
    model_layers = model.get_layers(args.model_type)
    model_history = model.train_and_evaluate(model_layers,args.epoch, args.steps_per_epoch, args.job_dir, args.nrows, args.loss_class_weighted, args.model_type)
    
if __name__=='__main__':
    main()
