


/××××××××××××××使用数据集命令××××××××××××××××××/

bin/dso_dataset \
		files=/home/sunlibo/Downloads/dataset/sequence_14/images \
		calib=/home/sunlibo/Downloads/dataset/sequence_14/camera.txt \
		gamma=/home/sunlibo/Downloads/dataset/sequence_14/pcalib.txt \
		vignette=/home/sunlibo/Downloads/dataset/sequence_14/vignette.png \
		preset=0 \
		mode=0


//使用KIT数据集

./dso_dataset files=/home/sunlibo/datasets/image_0/ calib=/home/sunlibo/datasets/image_0/camera.txt preset=2 mode=1



./bin/dso_dataset files=/home/sunlibo/datasets/image_0/ calib=/home/sunlibo/datasets/image_0/camera.txt preset=2 mode=1


/××××××××××××××使用实时摄像头命令××××××××××××××××××


./dso_dataset files=/home/sunlibo/Downloads/dataset/sequence_14/images / calib=/home/sunlibo/code/dso/useOwnLiveCamera/camera.txt preset=2 mode=1


./dso_dataset files=/home/sunlibo/Downloads/dataset/sequence_14/images / calib=/home/sunlibo/code/dso/useOwnLiveCamera/camera.txt mode=1