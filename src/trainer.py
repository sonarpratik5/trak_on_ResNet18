import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def train_model(net, trainloader, testloader, epochs=200, device='cuda', checkpoint_dir='checkpoints', resume=False):
    net = net.to(device)
    
    # SAFETY: Only init weights if we are NOT resuming/finetuning
    if not resume:
        net.apply(init_weights)
        print("✓ Weights initialized", flush=True)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    best_acc = 0.0
    start_epoch = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("="*80, flush=True)
    print(f"STARTING TRAINING: {epochs} EPOCHS", flush=True)
    print("="*80, flush=True)

    for epoch in range(start_epoch, epochs):
        net.train()
        running_loss = 0
        correct, total = 0, 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 0 and i > 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(trainloader)}], Loss: {running_loss/100:.4f}', flush=True)
                running_loss = 0.0

        train_acc = 100 * correct / total
        scheduler.step()

        # Evaluate
        net.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_acc = 100 * correct_test / total_test
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}")

        # Checkpoint Management
        checkpoint_path = os.path.join(checkpoint_dir, f"resnet_epoch_{epoch}.pt")
        torch.save({"model_state": net.state_dict()}, checkpoint_path)
        
        # FIXED: Sort by Integer Epoch Number (prevents "10 before 2" bug)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('resnet_epoch_')]
        # Extracts '5' from 'resnet_epoch_5.pt' and uses it to sort
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Keep only last 5 checkpoints
        if len(checkpoint_files) > 5:
            for old_file in checkpoint_files[:-5]:
                os.remove(os.path.join(checkpoint_dir, old_file))

        # Save Best Model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state": net.state_dict(),
                "test_acc": test_acc,
                "epoch": epoch
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"⭐ New best accuracy: {test_acc:.2f}% (saved)")

        # Periodic Memory Cleanup
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    print(f"Training Complete. Best Acc: {best_acc:.2f}%")