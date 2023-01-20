rem place in automatic 1111 webui direcory and run to backup main files
mkdir backup
for %%a in (embeddings\*) do xcopy "%%a" "backup\embeddings\"
for %%a in (config.json ui-config.json styles.csv webui-user.bat webui-user.sh webui-macos-env.sh) do xcopy "%%a" "backup"
