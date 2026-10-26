# 自动重连与断点下载

`download_sftp.py` 使用系统 OpenSSH SFTP，无额外 Python 依赖，沿用 `~/.ssh/config` 的主机别名、端口、密钥和跳板配置。网络断开后自动重连并 `reget`，不保证网络永不中断。

在 Windows PowerShell 中运行：

```powershell
cd C:\workplace\code\train_yolo
git pull --ff-only

python download_sftp.py 26080-yc /root/autodl-tmp/capt/code/train_yolo/animals_30k.zip `
  --output C:\workplace\code\train_yolo\datasets\animals_30k.zip `
  --retries 0
```

`--output` 是完整文件路径。现有 scp 下载的部分文件可继续使用，但必须来自同一份未变化的远程文件；远程 ZIP 不要重新生成或修改。已有错误前缀不能靠续传修复。不要同时运行两个传输程序写同一个文件。

- `--retries 0`：无限重试；默认 10 次额外重试。间隔从 10 秒逐渐增加，最长 60 秒。
- 保活请求每 15 秒发送，无响应达到 3 次后由 SSH 断开并重新连接。
- `--stall-timeout 600`：文件 10 分钟没有增长则重连；设为 0 关闭。每 10 秒显示本地已下载 MiB 和最近速度。
- Ctrl+C 停止并保留部分文件；重跑同一命令继续。
- 认证失败、路径错误等也会消耗重试次数；无限模式下出现这些错误应 Ctrl+C 后修正。

自动重连采用 BatchMode，需要 SSH 密钥或 ssh-agent，不能自动输入密码。先确认下面命令无需输入密码即可运行：

```powershell
ssh -o BatchMode=yes 26080-yc "echo ready"
```

如果未成功，先通过普通 `ssh 26080-yc` 确认主机身份并配置密钥登录；脚本不会绕过主机密钥检查，也不会保存密码。

## 完整性校验

先读取远程文件 SHA256：

```powershell
ssh 26080-yc "sha256sum /root/autodl-tmp/capt/code/train_yolo/animals_30k.zip"
```

可在下载命令追加 `--sha256 <上面输出的64位哈希>`，完成后自动读取本地完整文件校验。哈希不一致会报错并保留文件，不会用追加下载掩盖问题。未指定哈希时，只依据 SFTP 的成功退出判断传输完成，不额外声明文件哈希一致。

参考：[OpenSSH SFTP 的 reget 与批处理](https://man.openbsd.org/sftp)、[SSH 保活参数](https://man.openbsd.org/ssh_config)。
