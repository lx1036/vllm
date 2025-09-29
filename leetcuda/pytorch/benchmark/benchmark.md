



nx-infer-dev 的 gpu-2 机器上的测试结果： 


kata gpu pod

```md
root@kata-liuxiang-gpu-qemu7-6c576b7d8b-th6tx:/workspace# python3 benchmark.py 
GPU 0 -> GPU 1: 4.00 GB/s
GPU 0 -> GPU 2: 2.86 GB/s
GPU 0 -> GPU 3: 5.88 GB/s
GPU 0 -> GPU 4: 5.88 GB/s
GPU 0 -> GPU 5: 5.88 GB/s
GPU 1 -> GPU 0: 4.00 GB/s
GPU 1 -> GPU 2: 3.03 GB/s
GPU 1 -> GPU 3: 5.87 GB/s
GPU 1 -> GPU 4: 5.90 GB/s
GPU 1 -> GPU 5: 6.25 GB/s
GPU 2 -> GPU 0: 3.85 GB/s
GPU 2 -> GPU 1: 4.16 GB/s
GPU 2 -> GPU 3: 5.89 GB/s
GPU 2 -> GPU 4: 6.25 GB/s
GPU 2 -> GPU 5: 5.88 GB/s
GPU 3 -> GPU 0: 2.50 GB/s
GPU 3 -> GPU 1: 2.70 GB/s
GPU 3 -> GPU 2: 2.56 GB/s
GPU 3 -> GPU 4: 1.85 GB/s
GPU 3 -> GPU 5: 1.82 GB/s
GPU 4 -> GPU 0: 2.38 GB/s
GPU 4 -> GPU 1: 2.70 GB/s
GPU 4 -> GPU 2: 2.56 GB/s
GPU 4 -> GPU 3: 1.85 GB/s
GPU 4 -> GPU 5: 1.82 GB/s
GPU 5 -> GPU 0: 2.38 GB/s
GPU 5 -> GPU 1: 2.70 GB/s
GPU 5 -> GPU 2: 2.56 GB/s
GPU 5 -> GPU 3: 1.85 GB/s
GPU 5 -> GPU 4: 1.82 GB/s
```



gpu pod:

```md
root@gpu-perf2-76b5b5cdd8-tnh4x:/workspace# python3 benchmark-new.py 
GPU 0 -> GPU 1: 14.13 GB/s
GPU 0 -> GPU 2: 19.24 GB/s
GPU 0 -> GPU 3: 12.38 GB/s
GPU 0 -> GPU 4: 12.79 GB/s
GPU 0 -> GPU 5: 12.21 GB/s
GPU 1 -> GPU 0: 17.73 GB/s
GPU 1 -> GPU 2: 15391.94 GB/s
GPU 1 -> GPU 3: 20550.24 GB/s
GPU 1 -> GPU 4: 21388.60 GB/s
GPU 1 -> GPU 5: 21454.24 GB/s
GPU 2 -> GPU 0: 2.56 GB/s
GPU 2 -> GPU 1: 16461.16 GB/s
GPU 2 -> GPU 3: 27648.68 GB/s
GPU 2 -> GPU 4: 29683.68 GB/s
GPU 2 -> GPU 5: 29086.71 GB/s
GPU 3 -> GPU 0: 2.68 GB/s
GPU 3 -> GPU 1: 16100.98 GB/s
GPU 3 -> GPU 2: 28301.65 GB/s
GPU 3 -> GPU 4: 27685.17 GB/s
GPU 3 -> GPU 5: 29351.32 GB/s
GPU 4 -> GPU 0: 1.90 GB/s
GPU 4 -> GPU 1: 16953.53 GB/s
GPU 4 -> GPU 2: 30023.65 GB/s
GPU 4 -> GPU 3: 30066.70 GB/s
GPU 4 -> GPU 5: 29937.93 GB/s
GPU 5 -> GPU 0: 2.57 GB/s
GPU 5 -> GPU 1: 19178.34 GB/s
GPU 5 -> GPU 2: 30415.55 GB/s
GPU 5 -> GPU 3: 30459.72 GB/s
GPU 5 -> GPU 4: 30885.89 GB/s
```


nvbandwidth:
```md

```




