### tensorflow2.0搭建ResNet进行cifar识别任务

----repository说明----

time: 2019/6/20 coded by awei

email: [weizw.scut@gmail.com](mailto:weizw.scut@gmail.com)

school: scut(华南理工大学)



### resnet.py

代码中main()不做data augmentation,不作model save

main2()增加data augmentation并且每200step保存一个weights

main3()可载入已训练1000次的weights,需修改path

运行代码自动下载Cifar数据集,也可手动下载