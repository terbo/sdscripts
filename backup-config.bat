mkdir backup
for %%a in (embeddings\*) do xcopy "%%a" "backup\embeddings\"
for %%a in (config.json ui-config.json styles.csv webui-user.bat webui-user.sh webui-macos-env.sh) do xcopy "%%a" "backup"
