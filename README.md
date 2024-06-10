# Backdoor Attack with Sparse and Invisible Trigger

This is the official implementation of our paper "Backdoor Attack with Sparse and Invisible Trigger". This research project is developed based on Python 3 and Pytorch, created by Yinghua Gao and [Yiming Li](https://liyiming.tech/).

## Requirements

We have tested the code under the following environment settings:

- python = 3.7.10
- torch = 1.7.1
- torchvision = 0.8.2

## Quick Start

**Step 1: Train surrogate model**

In SIBA, we must train a surrogate model to optimize the trigger.

```
CUDA_VISIBLE_DEVICES=0 python train_surrogate_cifar.py --model resnet18 --save_surrogate save_surrogate --epochs 100
```

**Step 2: Optimize SIBA trigger**

With the trained surrogate model, we generate the trigger by an alternative optimization method.

```
CUDA_VISIBLE_DEVICES=0 python optimize_siba.py --surrogate_model resnet18 --save_surrogate save_surrogate --save_trigger save_trigger --y_target 0 --epochs 200 --k 100 --epsilon 8.0 --step_decay 0.8 --epoch_step 5
```
**Step 3: Train backdoored model**

With the optimized SIBA trigger, we train the backdoored model.

```
CUDA_VISIBLE_DEVICES=0 python train_poison_cifar.py --save_dir save_backdoor --save_trigger save_trigger --y_target 0 --epochs 100 --poison_rate 0.01
```

## Citation

If this work or our codes are useful for your research, please kindly cite our paper as follows.

```
@article{gao2024backdoor,
  title={Backdoor attack with sparse and invisible trigger},
  author={Gao, Yinghua and Li, Yiming and Gong, Xueluan and Li, Zhifeng and Xia, Shu-Tao and Wang Qian},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024}
}
```

## Acknowledgement

Our implementation is based on the following projects. We sincerely thank the authors for releasing their codes.

- [Universal Adversarial Perturbations on PyTorch](https://github.com/kenny-co/sgd-uap-torch)
- [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://github.com/csdongxian/ANP_backdoor)
