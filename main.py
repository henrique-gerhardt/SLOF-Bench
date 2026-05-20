from train import run
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_flow', type=int, default=300)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--train_batch', type=int, default=8)
    parser.add_argument('--val_batch', type=int, default=8)
    
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=int, default=1e-4)
    
    parser.add_argument('--clip_grad', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--use_density_mask', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--switch_rotation', action='store_true')
    parser.add_argument('--double_rotation', action='store_true')
    parser.add_argument('--grad_scaler', action='store_true')
    parser.add_argument('--ktn', action='store_true')
    
    
    
    parser.add_argument('--data_root', default='/data/keshav/flow360/FLOW360_train_test', help="path to train test data")
    
    parser.add_argument('--model_path', default='weights/singlerotation.pt', help="path to saved model")
    parser.add_argument('--csv_save', default='singlerotation', help="name of reports to be saved")
    parser.add_argument('--model_save', default='weights/jpt.pt', help="name of model to be saved")
    
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    
    
    args = parser.parse_args()
    
    run(args)
    
