# GitHub Actions 打包经验沉淀（audio_cutter）

> 来源：v6.3.0 ~ v6.3.4 连续 5 次 CI 构建与分发问题排查（2026-10-01 ~ 10-02）。
> 目的：下次改打包/加依赖时，先过一遍本文清单，避免重复踩坑。

## 一、踩过的坑（按时间线）

| 版本 | 症状 | 根因 | 修复 |
|---|---|---|---|
| v6.3.0 | 打包缺 silero_vad 依赖 | workflow 只装了 PySide6/numpy，没装 torch、silero-vad | 补齐 pip 依赖 |
| v6.3.1 | 运行缺模型文件 | 未收集 silero_vad 包内数据 | 加 `--collect-all silero_vad` |
| v6.3.2 | 用户报缺 silero_vad.jit | collect-all 在 CI 环境分析漏文件（静默缺，构建不报错） | 升级 collect 参数组合（实质靠 collect-all 补齐） |
| v6.3.3 | 用户报缺 silero_vad.jit（文件明明在包里） | **中文用户名**：torch.jit.load 对文件路径走 Windows ANSI fopen，`c:\users\黄芪\...` 解不出 → errno 2 | 改为 importlib.resources 读字节 + `torch.jit.load(BytesIO)`，不经过路径 |
| — | 误判为「缺 ffmpeg」曾打包 ffmpeg.exe | 多余 101MB 分发 | 移除分发，README 注明用户自备 ffmpeg |
| v6.3.3 → 6.3.4 | 排查 v6.3.3 用户报错时走了弯路：先怀疑旧版/下载损坏/杀软，三条全被用户否认 | **信息不足时臆测排查方向**：最初没让用户提供报错原文，也没注意路径里的中文用户名；还差点把 216MB exe 反复下载验尸，浪费时间 | 排查第一步永远是「要报错原文 + 发生时机」，先看路径有没有非 ASCII 字符（见教训 3、5） |
| v6.3.4 | 本地修复验证只能证「代码对」，证不了「全环境对」 | 中文用户名问题在本地（ASCII 用户名）**无法复现**，只能靠推理定位 + 用户机器实测确认 | 修复发版后必须请用户实测回执，才算闭环；不可自行宣布「已修复」 |

## 二、核心教训（必须记住）

### 1. 「CI 构建成功」≠「exe 能用」
PyInstaller 收集失败（缺 hidden-import、缺数据文件）**构建时不报错**，只在用户运行时炸。
→ 每次改完打包配置，必须**本地先打一次包 + 验证包内文件**再发版。

### 2. 本地验证方法（已验证有效）
```bash
.venv/bin/pyinstaller --onefile --windowed --name audio_cutter \
  --hidden-import torch \
  --collect-all silero_vad \
  --collect-data silero_vad \
  --collect-binaries silero_vad \
  --collect-all PySide6.QtMultimedia \
  audio_cutter.py
# 验证模型真的打进去了（TOC 级检查，非字符串扫描）：
python3 - <<'EOF'
import struct
f = open('dist/audio_cutter','rb'); f.seek(0,2); size=f.tell()
f.seek(size-88); magic, lp, toc, tl, *_ = struct.unpack('!8siiii64s', f.read(88))
f.seek(size-lp+toc); buf=f.read(tl); p=0
while p+18 <= len(buf):
    eLen, off, cs, us = struct.unpack_from('!iiii', buf, p)
    if eLen<=0: break
    name = buf[p+18:p+eLen].rstrip(b'\0').decode(errors='replace')
    if 'silero_vad' in name: print(name, us)
    p += eLen
EOF
```

### 3. 本地能跑 ≠ 所有用户机器能跑
- 我本地环境（Linux、ASCII 用户名、干净 .venv）验证不了**用户环境差异**。
- 典型环境差异：**中文/非 ASCII 用户名**（Temp 路径含中文）、无 ffmpeg、杀软。
- 修运行时 bug 时先问：用户的用户名/路径/系统语言是什么？要报错原文（含完整路径）。

### 4. 「errno 2 / no such file」不一定是文件不存在
Windows ANSI fopen 遇到 GBK 编不了的字符（中文用户名）就报 errno 2——**文件其实在**。
凡是库内部用窄字符 API 打开路径的（torch.jit.load 是重灾区），一律改成：
```python
from importlib import resources
import io, torch
b = resources.files("silero_vad.data").joinpath("silero_vad.jit").read_bytes()
model = torch.jit.load(io.BytesIO(b), map_location="cpu")  # 字节流，不碰路径
```

### 5. 排查流程（以后照此执行）
1. 用户报错 → 要**报错原文**（完整路径含用户名）+ 发生时机（启动/加载音频/点候选）；
2. 先看路径里有没有**非 ASCII 字符**（中文用户名是高发区）；
3. 确认用户版本 = 最新 Release（下载时间 vs 发布时间）；
4. 下载 Release exe 做二进制验尸（确认文件在不在包里），再决定改不改打包；
5. 修复后：本地打包 → TOC 验证 → 发版 → 下载 Release 包复验。

## 三、当前 workflow 的打包参数（勿随意删改）

```yaml
pyinstaller --onefile --windowed --name audio_cutter
  --hidden-import torch            # torch 动态导入，必须显式
  --collect-all silero_vad         # VAD 包整体
  --collect-data silero_vad        # .jit/.onnx 模型文件（缺了运行时 errno 2）
  --collect-binaries silero_vad
  --collect-all PySide6.QtMultimedia  # 播放器后端 DLL
```
- **ffmpeg 不打包不分发**：用户自备并加入 PATH（README 已写明）。
- 若未来加新依赖：先本地打包验证，再动 workflow。

## 四、版本记录

- v6.3.3：最后一版「路径加载」实现，中文用户名机器不可用（包本身完整）。
- v6.3.4：字节流加载修复版，**当前最新**；注意修复正确性仍需用户实测回执确认。
- 历史 tag：v6.3.0/6.3.1/6.3.2 的 exe 请勿分发（依赖/文件缺失）。
