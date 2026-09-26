# AudioCutter 音频波形裁切工具

对单条长音频进行分段打轴并导出为多个 MP3 的桌面工具（Windows 免安装 exe，或 Python 直接运行）。

## 功能

- **波形打轴**：加载音频 → Tab/按钮/右键菜单标记边界 → 拖动橙色虚线微调
- **右键菜单**（全中文）：空白处加边界、边界上删除（两侧段合并）、任意位置均可用
- **片段列表**：编号自动重排；勾选/取消决定是否导出；顶部实时计数「已选 X / 共 N 段」；选中段按 Del 删右侧边界；双击试听
- **撤销**：Ctrl+Z / 按钮，覆盖增删边界、勾选排除等操作（最多 100 步）
- **自动标记候选**（实验性）：Silero VAD 找出最可能是段落边界的位置（黄色虚线，默认 15 个可调），右键逐个「转为边界 / 移除」——候选仅供参考，需人工确认；VAD 失效音频自动降级能量兜底
- **导出**：按编号导出 MP3（ffmpeg），生成对应表.md；跳过段不占编号
- **项目保存/加载**：JSON 记录边界、排除状态、编号起点

## Windows 使用

从 [Releases](../../releases) 下载 `audiocutter.exe` 与 `ffmpeg.exe`，放同一目录，双击运行。

## 从源码运行

```bash
pip install PySide6 numpy
# 自动标记候选功能另需: pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install silero-vad
# 系统需安装 ffmpeg 并加入 PATH
python audio_cutter.py [音频文件] [故事文本.md]
```

## 打包

推 `v*` 标签或在 Actions 页手动触发 → GitHub Actions 自动编译 Windows exe 并发布 Release。

## 版本说明

- v6.3：勾选计数 / 右键删边界 / 自动候选边界（半自动）/ 撤销 / 拖动修复
- 全自动边界方案经实测废弃（BGM 停顿与真停顿声学不可分），详见开发过程结论：候选制是当前可靠形态
