import argparse
from src import (
    set_seed, 
    get_device, 
    ResNet18, 
    get_dataloaders, 
    train_model, 
    run_trak_analysis
)

def main():
    parser = argparse.ArgumentParser(description='ResNet TRAK Pipeline')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'analyze', 'all'], 
                        help='Mode: train, analyze (TRAK), or all')
    args = parser.parse_args()

    # 1. Setup
    SEED = 42
    set_seed(SEED)
    device = get_device()
    
    # 2. Data & Model
    trainloader, testloader, train_set, test_set, classes = get_dataloaders(seed=SEED)
    net = ResNet18(num_classes=10).to(device)

    # 3. Execution
    if args.mode in ['train', 'all']:
        train_model(net, trainloader, testloader, epochs=200, device=device)

    if args.mode in ['analyze', 'all']:
        run_trak_analysis(net, train_set, testloader, classes, 
                          checkpoint_path='checkpoints/best_model.pt', 
                          device=device, seed=SEED)

if __name__ == "__main__":
    main()


"""Train & Analyze: python main.py --mode all

Train Only: python main.py --mode train

Analyze Only: python main.py --mode analyze"""