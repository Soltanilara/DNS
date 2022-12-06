#!/bin/bash

var=1 #控制最外层while中断
while [ ${var} -eq 1 ]
  do
    i=$(nvidia-smi --id=3 --query-gpu=memory.used --format=csv,noheader,nounits)
    #nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits：查询GPU的内存使用信息，以CSV格式返回（无表头和单位）
    if [ ${i} -lt 2048 ] #如何GPU卡i的内存使用小于3000,两个[]有空格
    then
      echo 'GPU 3 is avaiable'  #向终端输出GPUi是有空闲
      sh train.sh
      var=0
      break
    else
      sleep 1m
    fi
  done