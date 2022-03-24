chmod +x dw_trainzip.sh 
chmod +x dw_resnet.sh
echo ">>> downloading train.zip"
./dw_trainzip.sh 
echo "<< downloaded train.zip"
echo ">>> downloading resnet"
./dw_resnet.sh
echo "<< downloaded resnet"
echo ">>> downloading test.zip"
./dw_testzip.sh
echo "<< downloaded test.zip"

echo ">>> unzipping train.zip"
! unzip -q train.zip
echo "<< unzipped train.zip"
echo ">>> unzipping test.zip"
! unzip -q test.zip
echo "<< unzipped test.zip"
rm train.zip
