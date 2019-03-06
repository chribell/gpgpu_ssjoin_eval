# An Empirical Evaluation of Exact Set Similarity Join Techniques Using GPUs
The goal of this work is to evaluate the GPU techniques proposed for exact set similarity join and compare them to the best CPU standalone alternative.


## Abstract 
Exact set similarity join is a notoriously expensive operation, for which several solutions have been proposed. Recently, there have been studies that present a comparative analysis using MapReduce or a non-parallel setting. Our contribution is that we complement these works through conducting a thorough evaluation of the state-of-the-art GPU-enabled techniques. These techniques are highly diverse in their key features and our experiments manage to reveal the key strengths of each one. As we explain, in real-life applications, there is neither a dominating solution nor a solution without its own sweet spot including not using the GPU at all. All our work is repeatable and extensible.

## Techniques
CPU standalone: http://ssjoin.dbresearch.uni-salzburg.at/

CPU-GPU: https://github.com/chribell/gpussjoin/

gSSJoin: this repo

fgSSJoin: this repo

sf-gSSJoin: this repo

bitmap: this repo

## Building
```
cd (gssjoin|fgssjoin|sf-gssjoin|bitmap) && mkdir release && cd release
#for compute capability 6.1
CC=gcc-5 CXX=g++-5 cmake -DCMAKE_BUILD_TYPE=Release -DSM_ARCH=61 CMAKE_CXX_FLAGS=-m64 .. 
```

## Execution

### gSSJoin

```
./gssjoin <input_file> <threshold> <output_file> <number_of_gpus> <aggregate>
```
```
  <input_file>       file, each line a record
  <threshold>       normalized threshold
  <output_file>      file to output pairs (if not aggregate)
  <number_of_gpus>  the number of gpus to be involved in the set similarity join
  <aggregate>       flag if to perform an aggregation on top of the join (0 | 1)
```

### fgSSJoin

```
./fgssjoin <input_file> <threshold> <output_file> <number_of_gpus> <size of blocks> <aggregate>
```
```
  <input_file>       file, each line a record
  <threshold>       normalized threshold
  <output_file>      file to output pairs (if not aggregate)
  <number_of_gpus>  the number of gpus to be involved in the set similarity join
  <size of blocks>  the size of each input collection partition block
  <aggregate>       flag if to perform an aggregation on top of the join (0 | 1)
```

### sf-gSSJoin

```
./sfgssjoin <input_file> <threshold> <output_file> <number_of_gpus> <size of blocks> <aggregate>
```
```
  <input_file>       file, each line a record
  <threshold>       normalized threshold
  <output_file>      file to output pairs (if not aggregate)
  <number_of_gpus>  the number of gpus to be involved in the set similarity join
  <size of blocks>  the size of each input collection partition block
  <aggregate>       flag if to perform an aggregation on top of the join (0 | 1)
```

### bitmap

```
./bitmap --input <input_file> --threshold <threshold> --bitmap <bitmap>
```

```
  <input_file>       file, each line a record
  <threshold>       normalized threshold
  <bitmap>          bitmap signature size (bits)
```

## Datasets
The datasets and the preprocess scripts can be found at http://ssjoin.dbresearch.uni-salzburg.at/datasets.html
