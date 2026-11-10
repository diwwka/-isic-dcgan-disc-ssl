$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location A:\MasterThesis\code
$R = "A:\MasterThesis\results"
if (-not (Test-Path A:\MasterThesis\cache\isic2019_128.npy)) {
    python -u preprocess.py D:\isic A:\MasterThesis\cache 128 2>&1 | Tee-Object -FilePath "$R\log_pre128.txt"
}
foreach ($s in 1,2,3,4) {
    python -u exp.py train --model resnet_imagenet --frac 100 --seed $s --size 224 --bs 32 --amp --tag ref224 2>&1 | Tee-Object -FilePath "$R\log_ref224_s$s.txt"
}
python -u exp.py all --seeds 0 1 2 --size 128 --out A:\MasterThesis\results128 --models disc_scratch disc_ft resnet_imagenet --fracs 100 10 2>&1 | Tee-Object -FilePath "A:\MasterThesis\log_128.txt"
Write-Host "ГОТОВО"
