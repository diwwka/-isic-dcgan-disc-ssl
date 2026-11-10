# Запускати в PowerShell з папки isic-dcgan-disc-ssl, ПІСЛЯ того як ти скопіював решту файлів у src/ і results/
# 1) Створи ПУСТИЙ репозиторій на github.com (New repository, без README/LICENSE), назву візьми isic-dcgan-disc-ssl
# 2) Підстав свій логін:
$user = "ТВІЙ_ЛОГІН"
git init -b main
git add .
git status          # переглянь: не має бути .npy/.pth/.png із даними ISIC
git commit -m "Code and results for master's thesis"
git remote add origin "https://github.com/$user/isic-dcgan-disc-ssl.git"
git push -u origin main
