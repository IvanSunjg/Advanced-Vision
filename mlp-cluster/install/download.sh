chmod +x dw_trainzip.sh 
chmod +x dw_resnet.sh
echo ">>> downloading train.zip"
./dw_trainzip.sh 
echo "<< downloaded train.zip"
echo ">>> downloading resnet"
./dw_resnet.sh
echo "<< downloaded resnet"
echo ">>> unzipping train.zip"
! unzip train.zip
echo "<< unzipped train.zip"
rm train.zip
