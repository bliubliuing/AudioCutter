#!/usr/bin/env python3
"""音频可视化裁切工具（手动打轴版）

波形显示 + 故事文本同步 + Tab 标记边界 + 拖动微调 + 按编号导出 MP3。

运行: .venv/bin/python audio_cutter.py [音频文件] [故事文本md]
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QUrl, QPointF, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)

pg.setConfigOptions(antialias=False, background="#fafafa", foreground="#333")


def decode_pcm(path: str) -> tuple[np.ndarray, int]:
    """ffmpeg 解码为单声道 float32，统一重采样到 22050Hz，返回 (samples, sr)。"""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "22050",
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    pcm = np.frombuffer(raw, dtype=np.float32)
    return pcm, 22050


class ExportWorker(QThread):
    """后台导出线程：逐段 ffmpeg 切分，避免界面无响应。

    信号:
      progress(idx, seg_id, state)  state: 'run' | 'ok' | 'fail'
      done(ok_count, skip_count, out_dir, errors)
    """

    progress = Signal(int, str, str)
    done = Signal(int, int, str, list)

    def __init__(self, audio_path, segs, excluded, text_segments, out_dir, start_id):
        super().__init__()
        self.audio_path = audio_path
        self.segs = segs              # [(start, end, seg_idx)]
        self.excluded = excluded      # set[int]
        self.text_segments = text_segments
        self.out_dir = out_dir
        self.start_id = start_id
        self.cancelled = False

    def run(self):
        out = Path(self.out_dir)
        audio_name = Path(self.audio_path).name
        rows = [f"源音频：{audio_name}",
                "",
                "| 贴纸编号 | 状态 | 时间区间 | 文本 |", "|---|---|---|---|"]
        next_id = self.start_id
        ok = skipped = 0
        errors = []
        for idx, (s, e, i) in enumerate(self.segs):
            if self.cancelled:
                break
            txt = self.text_segments[i] if i < len(self.text_segments) else ""
            if i in self.excluded:
                rows.append(f"| — | 跳过 | {ExportWorker._fmt(s)} ~ {ExportWorker._fmt(e)} | {txt} |")
                skipped += 1
                self.progress.emit(idx, "—", "skip")
                continue
            seg_id = f"{next_id:05d}"
            next_id += 1
            dst = out / f"{seg_id}.mp3"
            self.progress.emit(idx, seg_id, "run")
            r = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}",
                 "-t", f"{max(0.1, e - s):.3f}",
                 "-i", self.audio_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                 str(dst)], capture_output=True)
            if r.returncode == 0:
                ok += 1
                self.progress.emit(idx, seg_id, "ok")
            else:
                dst.unlink(missing_ok=True)
                errors.append(f"{seg_id}: {r.stderr.decode()[-200:]}")
                self.progress.emit(idx, seg_id, "fail")
            rows.append(f"| {seg_id} | 正常 | {ExportWorker._fmt(s)} ~ {ExportWorker._fmt(e)} | {txt} |")
        (out / "对应表.md").write_text("\n".join(rows), encoding="utf-8")
        self.done.emit(ok, skipped, str(out), errors)

    @staticmethod
    def _fmt(t):
        t = max(0, t)
        return f"{int(t // 60)}:{t % 60:04.1f}"


class CandidateWorker(QThread):
    """后台 VAD 候选边界线程：v6.3 半自动版——只找候选+排序，不自动采用"""
    done = Signal(list, str)          # (候选时间点列表, 模式说明)
    failed = Signal(str)

    def __init__(self, audio_path, n_candidates=15):
        super().__init__()
        self.audio_path = audio_path
        self.n = n_candidates

    @staticmethod
    def _load_wav_16k(path):
        """ffmpeg 解码为 16k 单声道 float32（Silero 要求）"""
        import subprocess as sp
        r = sp.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar",
                    "16000", "-f", "wav", "-"], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode()[-200:])
        import wave, io
        with wave.open(io.BytesIO(r.stdout), 'rb') as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0

    def run(self):
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as e:
            self.failed.emit(f"缺少依赖 silero-vad/torch: {e}")
            return
        try:
            x = self._load_wav_16k(self.audio_path)
            # 用字节流加载 jit 模型：torch.jit.load 对文件路径走 ANSI fopen，
            # 中文用户名（如 c:\users\黄芪\...）会导致 errno 2，BytesIO 则无此问题
            import io
            from importlib import resources
            jit_bytes = resources.files("silero_vad.data").joinpath("silero_vad.jit").read_bytes()
            model = torch.jit.load(io.BytesIO(jit_bytes), map_location="cpu")
            # 逐帧概率：30ms 窗 / 10ms hop（10ms 网格）
            hop = 160
            probs = []
            t = torch.from_numpy
            for i in range(0, len(x) - 512 + 1, hop):
                with torch.no_grad():
                    probs.append(model(t(x[i:i + 512]), 16000).item())
            probs = np.array(probs, dtype=np.float32)

            mode = "vad"
            if (probs < 0.1).mean() > 0.9:
                # VAD 失效兜底：能量滞回（16k 上做能量）
                mode = "能量兜底（VAD 失效）"
                hop_e = 160
                frames = np.lib.stride_tricks.sliding_window_view(x, 512)[::hop_e]
                en_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
                n = min(len(probs), len(en_db))
                floor = np.percentile(en_db[:n], 10)
                sp = np.zeros(n, dtype=bool)
                s = False
                for i in range(n):
                    if not s and en_db[i] > floor + 8:
                        s = True
                    elif s and en_db[i] < floor + 5:
                        s = False
                    sp[i] = s
            else:
                # 滞回：进静音 p<0.3，出静音 p>0.5
                sp = np.zeros(len(probs), dtype=bool)
                s = False
                for i, p in enumerate(probs):
                    if s and p < 0.3:
                        s = False
                    elif not s and p > 0.5:
                        s = True
                    sp[i] = s

            # 静音候选段 ≥0.3s；无桥接（v6.3）
            sil = ~sp
            d = np.diff(sil.astype(int))
            starts = list(np.where(d == 1)[0] + 1)
            ends = list(np.where(d == -1)[0] + 1)
            if sil[0]:
                starts = [0] + starts
            if sil[-1]:
                ends = ends + [len(sil)]
            cands = [(a, b) for a, b in zip(starts, ends) if (b - a) >= 30]
            if not cands:
                self.done.emit([], mode)
                return

            # 评分：时长甜点 + 两侧 ±3s 人声回归度（仅排序）
            scored = []
            W = 300
            for a, b in cands:
                dur = (b - a) / 100
                if dur <= 0.5:
                    dsh = dur / 0.5
                elif dur <= 2.5:
                    dsh = 1.0
                else:
                    dsh = max(0.0, 1.0 - (dur - 2.5) / 2.5)
                pre = probs[max(0, a - W):a]
                post = probs[b:b + W]
                sur = 0.5 * pre.mean() + 0.5 * post.mean() if len(pre) and len(post) else 0.0
                pd = abs(probs[max(0, a - 50):a].mean() - probs[b:b + 50].mean())
                s = 0.55 * sur + 0.30 * dsh + 0.15 * min(pd * 2, 1.0)
                # 边界点：≤2s 用中点，>2s 用概率最低点
                if dur <= 2.0:
                    tpt = (a + b) / 2 / 100
                else:
                    tpt = (a + int(np.argmin(probs[a:b]))) / 100
                scored.append((s, tpt))
            scored.sort(reverse=True)

            # NMS：间隔 ≥4s，取前 N 个
            picked = []
            for s, tpt in scored:
                if all(abs(tpt - p) >= 4.0 for p in picked):
                    picked.append(tpt)
                    if len(picked) >= self.n:
                        break
            self.done.emit(sorted(picked), mode)
        except Exception as e:
            self.failed.emit(str(e))


class WaveformPlot(pg.PlotWidget):
    """波形 + 边界线 + 播放光标。emits: play_at(t), boundary_dragged(idx, t), boundary_delete(idx)"""

    def __init__(self):
        super().__init__()
        self.setMouseEnabled(x=True, y=False)
        self.showGrid(x=True, y=False, alpha=0.2)
        # 禁用 pyqtgraph 默认英文右键菜单（必须用 setMenuEnabled：
        # 直接置 menu=None 会残留 enableMenu=True，右键事件被 ViewBox 吞掉）
        _pi = self.getPlotItem()
        _pi.vb.setMenuEnabled(False)
        _pi.ctrlMenu = None
        self.vb = self.getPlotItem().vb
        self.vb.setMouseEnabled(x=True, y=False)

        self.bars: list[pg.InfiniteLine] = []
        self.cand_bars: list[pg.InfiniteLine] = []
        self.cursor = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen("#e74c3c", width=2))
        self.addItem(self.cursor)
        self.play_at = lambda t: None
        self.boundary_dragged = lambda i, t: None
        self.boundary_delete = lambda i: None
        self.boundary_add = lambda t: None
        self.candidate_clear_all = lambda: None

        # 鼠标左键点击波形即定位播放（拖动边界线由 InfiniteLine 自己接管）
        self.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, ev):
        if ev.double():
            return
        pos = ev.scenePos()
        if not self.vb.sceneBoundingRect().contains(pos):
            return
        t = self.vb.mapSceneToView(pos).x()
        if ev.button() == Qt.MouseButton.LeftButton:
            if t < 0:
                t = 0
            self.play_at(t)
        elif ev.button() == Qt.MouseButton.RightButton:
            self._on_right_click(t, ev)

    def _on_right_click(self, t, ev):
        """右键：候选线附近 → 转为边界/移除候选；正式边界附近 → 删除该边界"""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        actions = []
        # 候选命中（±0.5s）
        cand_idx = None
        for i, b in enumerate(self.cand_bars):
            if abs(b.value() - t) <= 0.5:
                cand_idx = i
                break
        if cand_idx is not None:
            cv = self.cand_bars[cand_idx].value()
            actions.append(("adopt", menu.addAction(f"将候选转为边界（{cv:.1f}s）")))
            actions.append(("drop", menu.addAction("移除此候选")))
        # 正式边界命中
        idx = self._nearest_boundary(t)
        if idx is not None:
            actions.append(("del", menu.addAction(f"删除此边界（{self.bars[idx].value():.1f}s）")))
        # 空白处兜底：任意右键都有菜单（"在此处加边界"是常用操作）
        actions.append(("add", menu.addAction(f"在此处添加边界（{t:.1f}s）")))
        if self.cand_bars:
            actions.append(("clearc", menu.addAction("清除全部候选")))
        # 用 QCursor.pos()：ev.screenPos() 返回 QPointF，传给 exec 会因签名
        # 不匹配抛异常且被事件系统吞掉，表现为"右键无反应"
        from PySide6.QtGui import QCursor
        chosen = menu.exec(QCursor.pos())
        for key, act in actions:
            if chosen == act:
                if key == "adopt":
                    self.candidate_adopt(self.cand_bars[cand_idx].value())
                elif key == "drop":
                    self.candidate_drop(cand_idx)
                elif key == "del":
                    self.boundary_delete(idx)
                elif key == "add":
                    self.boundary_add(t)
                elif key == "clearc":
                    self.candidate_clear_all()
                return

    def _nearest_boundary(self, t):
        """返回距离 t 最近的边界索引；若 >0.5s 视为未命中"""
        if not self.bars:
            return None
        best_i, best_d = None, 1e9
        for i, b in enumerate(self.bars):
            d = abs(b.value() - t)
            if d < best_d:
                best_i, best_d = i, d
        return best_i if best_d <= 0.5 else None

    def set_wave(self, pcm, sr):
        self.sr = sr
        # 降采样包络：每 bin 取 min/max
        bin_size = max(1, len(pcm) // 200000)
        n = len(pcm) // bin_size
        seg = pcm[: n * bin_size].reshape(n, bin_size)
        env_hi = seg.max(axis=1)
        env_lo = seg.min(axis=1)
        xs = np.arange(n) * bin_size / sr
        # 重画波形会 self.clear() 移除所有线条，先同步清空两个列表引用
        self.bars = []
        self.cand_bars = []
        self.removeItem(self.cursor)
        self.clear()
        self.addItem(self.cursor)
        self.wave_hi = pg.PlotCurveItem(xs, env_hi, pen=pg.mkPen("#5b8db8", w=1))
        self.wave_lo = pg.PlotCurveItem(xs, env_lo, pen=pg.mkPen("#5b8db8", w=1))
        self.addItem(self.wave_hi)
        self.addItem(self.wave_lo)
        self.setXRange(0, len(pcm) / sr, padding=0)

    def clear_bars(self):
        for b in self.bars:
            self.removeItem(b)
        self.bars = []
        # 注意：不清候选（cand_bars）——_refresh_segments 每次增删边界都会
        # 走 set_boundaries→clear_bars，若在此清候选，加一个正式边界候选就全没了

    def clear_candidates(self):
        for b in self.cand_bars:
            self.removeItem(b)
        self.cand_bars = []

    def set_candidates(self, times):
        """黄色细虚线候选边界（建议语义，不自动采用）"""
        self.clear_candidates()
        for t in times:
            bar = pg.InfiniteLine(
                pos=t, angle=90, movable=False,
                pen=pg.mkPen("#f1c40f", width=1, style=Qt.DashLine))
            self.addItem(bar)
            self.cand_bars.append(bar)

    def set_boundaries(self, times, duration):
        self.clear_bars()
        for i, t in enumerate(times):
            bar = pg.InfiniteLine(
                pos=t, angle=90, movable=True,
                pen=pg.mkPen("#e67e22", width=2, style=Qt.DashLine))
            bar.setBounds([0, duration])
            # 拖拽中：只把新位置传给数据层，不重建波形线（重建会销毁正在
            # 拖拽的线，导致拖拽中断、边界"消失"）
            bar.sigDragged.connect(lambda b, i=i: self.boundary_dragging(i, b.value()))
            # 拖拽结束：才整体刷新（重排序、重建列表与波形线）
            bar.sigPositionChangeFinished.connect(
                lambda b, i=i: self.boundary_dragged(i, b.value()))
            self.addItem(bar)
            self.bars.append(bar)

    def set_cursor(self, t):
        self.cursor.setPos(t)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音频裁切工具 · 点读贴制作")
        self.resize(1150, 700)

        self.sr = 22050
        self.duration = 0.0
        self.audio_path: str | None = None
        self.text_segments: list[str] = []
        # boundaries[i] = 时间点；len == len(segments)-1
        self.boundaries: list[float] = []
        self.project_path: str | None = None
        self._excluded: set[int] = set()  # 不导出的段索引
        self._undo_stack: list[tuple] = []  # 撤销快照: (boundaries副本, excluded副本)

        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setAudioOutput(self.audio_out)
        self.audio_out.setVolume(1.0)
        self.player.positionChanged.connect(self._on_pos)
        self.player.mediaStatusChanged.connect(self._on_status)

        self._build_ui()
        self._shortcuts()

        args = sys.argv[1:]
        if len(args) >= 1 and Path(args[0]).is_file():
            self.load_audio(args[0])
        if len(args) >= 2 and Path(args[1]).is_file():
            self.load_text(Path(args[1]))

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # 工具栏
        tb = QToolBar()
        for label, slot in [
            ("打开音频", self.open_audio), ("加载文本", self.open_text),
            ("打开项目", self.open_project), ("保存项目", self.save_project)]:
            a = QAction(label, self)
            a.triggered.connect(slot)
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" 起始编号 "))
        self.id_edit = QLineEdit("00201")
        self.id_edit.setFixedWidth(70)
        tb.addWidget(self.id_edit)
        b_export = QAction("导出 MP3 和对应表", self)
        b_export.triggered.connect(self.export)
        tb.addAction(b_export)
        self.btn_export = b_export  # 导出期间禁用
        self.addToolBar(tb)

        # 波形
        self.wave = WaveformPlot()
        self.wave.play_at = self._seek
        self.wave.boundary_dragged = self._boundary_dragged
        self.wave.boundary_delete = self._delete_boundary_at
        self.wave.candidate_adopt = self._adopt_candidate
        self.wave.candidate_drop = self._drop_candidate
        self.wave.boundary_add = self._add_boundary_at
        self.wave.candidate_clear_all = self.clear_all_candidates

        # 右侧：文本 + 片段列表
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        f = QFont()
        f.setPointSize(13)
        self.text_view.setFont(f)

        self.seg_list = QListWidget()
        self.seg_list.itemDoubleClicked.connect(self._play_segment)
        self.seg_list.itemChanged.connect(self._on_seg_checked)

        # 勾选计数标签（v6.3 功能1）——挂在实际使用的 rl 布局上
        self.seg_count_label = QLabel("")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.wave)
        right_wrap = QWidget()
        rl = QVBoxLayout(right_wrap)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("故事文本（当前段高亮）"))
        rl.addWidget(self.text_view, 2)
        rl.addWidget(QLabel("片段列表（双击试听该段）"))
        rl.addWidget(self.seg_count_label)
        rl.addWidget(self.seg_list, 1)
        split.addWidget(right_wrap)
        split.setSizes([760, 380])

        # 底部控制条
        bar = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_tab = QPushButton("标记边界 (Tab)")
        btn_tab.clicked.connect(self.mark_boundary)
        btn_del = QPushButton("删除最后边界")
        btn_del.clicked.connect(self.delete_last)
        btn_undo = QPushButton("撤销 (Ctrl+Z)")
        btn_undo.clicked.connect(self.undo)
        btn_prev = QPushButton("◀ 上一段")
        btn_prev.clicked.connect(lambda: self.jump_segment(-1))
        btn_next = QPushButton("下一段 ▶")
        btn_next.clicked.connect(lambda: self.jump_segment(1))
        # 自动标记候选（v6.3 半自动版：只标候选，不自动采用）
        self.btn_auto = QPushButton("自动标记候选")
        self.btn_auto.clicked.connect(self.auto_mark_candidates)
        self.cand_spin = QSpinBox()
        self.cand_spin.setRange(5, 30)
        self.cand_spin.setValue(15)
        self.cand_spin.setPrefix("候选数 ")
        self.cand_spin.setToolTip("自动标记的候选边界数量（5–30，默认 15）")
        self.btn_clear_cand = QPushButton("清除候选")
        self.btn_clear_cand.clicked.connect(self.clear_all_candidates)
        self.btn_clear_cand.setToolTip("移除波形上全部未转正的候选边界线")
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("font-family:monospace;")
        for w in (self.btn_play, btn_tab, btn_del, btn_undo, btn_prev, btn_next,
                  self.btn_auto, self.cand_spin, self.btn_clear_cand):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.time_label)

        root.addWidget(split, 1)
        root.addLayout(bar)
        self.setCentralWidget(central)

    def _shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_Tab), self, activated=self.mark_boundary)
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_selected_segment_boundary)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._nudge(-0.5))

    # ---------- 加载 ----------
    def open_audio(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "打开音频", "", "音频 (*.mp3 *.wav *.wma *.m4a *.flac)")
        if f:
            self.load_audio(f)

    def load_audio(self, path):
        try:
            pcm, sr = decode_pcm(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"解码失败: {e}")
            return
        self.audio_path = path
        self.sr = sr
        self.duration = len(pcm) / sr
        self.wave.set_wave(pcm, sr)
        self.player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))
        self.setWindowTitle(f"音频裁切 · {Path(path).name}")
        self.statusBar().showMessage(
            f"已加载 {path}  时长 {self._fmt(self.duration)}")

    def open_text(self):
        f, _ = QFileDialog.getOpenFileName(self, "加载故事文本", "", "文本 (*.md *.txt)")
        if f:
            self.load_text(Path(f))

    def load_text(self, path: Path):
        raw = path.read_text(encoding="utf-8")
        segs = [s.strip() for s in raw.split("\n---\n") if s.strip()]
        if len(segs) <= 1:  # 无 --- 分隔则整段
            segs = [s.strip() for s in raw.split("\n\n") if s.strip()]
        self.text_segments = segs
        html = []
        for i, s in enumerate(segs):
            html.append(f'<p id="seg{i}" style="margin:10px 0;">{s}</p>')
        self.text_view.setHtml("".join(html))
        self._refresh_segments()

    # ---------- 打轴 ----------
    # ---------- 撤销 ----------
    def _push_undo(self):
        """在改动 boundaries/_excluded 之前调用，保存当前状态快照"""
        self._undo_stack.append(
            (list(self.boundaries), set(self._excluded)))
        if len(self._undo_stack) > 100:  # 防无限增长
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            self.statusBar().showMessage("没有可撤销的操作")
            return
        boundaries, excluded = self._undo_stack.pop()
        self.boundaries = boundaries
        self._excluded = excluded
        self._refresh_segments()
        self.statusBar().showMessage(
            f"已撤销（剩余 {len(self._undo_stack)} 步可撤销）")

    def mark_boundary(self):
        if not self.audio_path:
            return
        t = self.player.position() / 1000.0
        self._push_undo()
        self.boundaries.append(t)
        self.boundaries.sort()
        self._refresh_segments()
        self.statusBar().showMessage(f"边界 {len(self.boundaries)}: {self._fmt(t)}")

    def delete_last(self):
        if self.boundaries:
            self._push_undo()
            self.boundaries.pop()
            self._refresh_segments()

    def _delete_boundary_at(self, idx):
        """删除指定索引的边界（波形右键菜单触发）"""
        if 0 <= idx < len(self.boundaries):
            self._push_undo()
            t = self.boundaries.pop(idx)
            self._refresh_segments()
            self.statusBar().showMessage(f"已删除边界 {self._fmt(t)}（两侧段已合并）")

    def _add_boundary_at(self, t):
        """在指定时间点添加边界（波形右键空白处菜单触发）"""
        t = max(0.0, t)
        self._push_undo()
        self.boundaries.append(t)
        self.boundaries.sort()
        self._refresh_segments()
        self.statusBar().showMessage(f"已添加边界 {self._fmt(t)}")

    def _delete_selected_segment_boundary(self):
        """Del 键：删除片段列表中选中段右侧的边界"""
        item = self.seg_list.currentItem()
        if item is None:
            return
        _, e, i = item.data(Qt.UserRole)
        # 找到该段终点对应的边界（容差 0.05s）
        for idx, t in enumerate(self.boundaries):
            if abs(t - e) < 0.05:
                self._delete_boundary_at(idx)
                return
        self.statusBar().showMessage("该段右侧无边界（末段）")

    # ---------- 自动标记候选边界（v6.3 半自动） ----------
    def auto_mark_candidates(self):
        if not self.audio_path:
            QMessageBox.warning(self, "提示", "请先加载音频。")
            return
        self.btn_auto.setEnabled(False)
        self.statusBar().showMessage("正在分析音频生成候选边界…（VAD 推理中，请稍候）")
        self._cand_worker = CandidateWorker(self.audio_path, self.cand_spin.value())
        self._cand_worker.done.connect(self._on_candidates_done)
        self._cand_worker.failed.connect(self._on_candidates_failed)
        self._cand_worker.start()

    def _on_candidates_done(self, times, mode):
        self.btn_auto.setEnabled(True)
        self.wave.set_candidates(times)
        self.statusBar().showMessage(
            f"已生成 {len(times)} 个候选边界（黄色虚线，模式：{mode}）。"
            "候选仅供参考，请逐个确认：右键候选附近可转为正式边界或忽略。")

    def _on_candidates_failed(self, msg):
        self.btn_auto.setEnabled(True)
        QMessageBox.critical(self, "自动标记失败", msg)

    def _adopt_candidate(self, t):
        """将候选转为正式边界，并移除该候选线"""
        self._push_undo()
        self.boundaries.append(t)
        self.boundaries.sort()
        self.wave.cand_bars = [b for b in self.wave.cand_bars
                               if abs(b.value() - t) > 0.01]
        self._refresh_segments()
        self.statusBar().showMessage(f"已采用候选边界 {self._fmt(t)}")

    def _drop_candidate(self, idx):
        """移除一个候选（不转正式边界）"""
        if 0 <= idx < len(self.wave.cand_bars):
            self.wave.removeItem(self.wave.cand_bars.pop(idx))
            self.statusBar().showMessage("已移除该候选")

    def clear_all_candidates(self):
        """一键清除波形上全部未转正的候选边界线"""
        n = len(self.wave.cand_bars)
        self.wave.clear_candidates()
        self.statusBar().showMessage(f"已清除 {n} 个候选边界")

    def _boundary_dragging(self, idx, t):
        """拖拽进行中：只更新数据值，不重建波形/列表（防拖拽中的线被销毁）"""
        if 0 <= idx < len(self.boundaries):
            self.boundaries[idx] = t

    def _boundary_dragged(self, idx, t):
        if 0 <= idx < len(self.boundaries):
            # 拖拽结束：更新值并整体刷新（排序、重建列表与波形线）
            self.boundaries[idx] = t
            self.boundaries.sort()
            self._refresh_segments()

    def _nudge(self, dt):
        """微调最后选中的边界（简化：最后一条）。"""
        if self.boundaries:
            self.boundaries[-1] = max(0, self.boundaries[-1] + dt)
            self._refresh_segments()

    # ---------- 播放 ----------
    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play.setText("播放")
        else:
            self.player.play()
            self.btn_play.setText("暂停")

    def _seek(self, t):
        self.player.setPosition(int(t * 1000))

    def _on_pos(self, ms):
        t = ms / 1000.0
        self.wave.set_cursor(t)
        self.time_label.setText(f"{self._fmt(t)} / {self._fmt(self.duration)}")
        self._highlight_current(t)

    def _on_status(self, st):
        if st == QMediaPlayer.EndOfMedia:
            self.btn_play.setText("播放")

    def _seg_times(self):
        """返回 [(start, end, seg_idx)]"""
        edges = [0.0] + sorted(self.boundaries) + [self.duration]
        out = []
        for i in range(len(edges) - 1):
            out.append((edges[i], edges[i + 1], i))
        return out

    def _highlight_current(self, t):
        for s, e, i in self._seg_times():
            if s <= t < e:
                self._mark_seg(i)
                break

    def _mark_seg(self, idx):
        n = len(self.text_segments)
        if not n:
            return
        html = []
        for i, s in enumerate(self.text_segments):
            if i == idx % n:
                html.append(f'<p style="background:#ffe680; font-weight:bold; margin:10px 0;">▶ {s}</p>')
            else:
                html.append(f'<p style="margin:10px 0;">{s}</p>')
        self.text_view.setHtml("".join(html))

    def _play_segment(self, item: QListWidgetItem):
        s, e, _ = item.data(Qt.UserRole)
        self._seek(s)
        self.player.play()
        self.btn_play.setText("暂停")
        # 到终点自动停（简单定时）
        from PySide6.QtCore import QTimer
        QTimer.singleShot(int((e - s) * 1000) + 300, self._stop_if_playing)

    def _stop_if_playing(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play.setText("播放")

    def jump_segment(self, d):
        cur = self.player.position() / 1000.0
        times = [(s, e, i) for s, e, i in self._seg_times()]
        if not times:
            return
        for s, e, i in times:
            if s <= cur < e:
                nxt = max(0, min(len(times) - 1, i + d))
                self._seek(times[nxt][0])
                self._mark_seg(times[nxt][2])
                return

    # ---------- 片段列表 / 导出 ----------
    def _refresh_segments(self):
        self.wave.set_boundaries(sorted(self.boundaries), self.duration)
        cur_sel = self._excluded.copy()
        self.seg_list.clear()
        n = len(self.text_segments)
        base_id = self.id_edit.text().strip() or "00001"
        try:
            start = int(base_id)
        except ValueError:
            start = 1
        next_id = start
        for s, e, i in self._seg_times():
            excluded = i in cur_sel
            if excluded:
                seg_id = "跳过"
            else:
                seg_id = f"{next_id:05d}"
                next_id += 1
            txt = (self.text_segments[i][:24] + "…"
                   if i < n and len(self.text_segments[i]) > 24
                   else (self.text_segments[i] if i < n else ""))
            item = QListWidgetItem(f"{seg_id}  {self._fmt(s)} ~ {self._fmt(e)}  {txt}")
            item.setData(Qt.UserRole, (s, e, i))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if excluded else Qt.Checked)
            if excluded:
                f = item.font()
                f.setStrikeOut(True)
                item.setFont(f)
                item.setForeground(Qt.gray)
            self.seg_list.addItem(item)
        # 勾选计数：X=勾选数(实际导出数)，N=总段数，Y=排除数
        total = len(self._seg_times())
        excluded = len(self._excluded)
        checked = total - excluded
        self.seg_count_label.setText(
            f"片段列表：已选 {checked} / 共 {total} 段（已排除 {excluded} 段不占编号）")

    def _on_seg_checked(self, item):
        _, _, i = item.data(Qt.UserRole)
        self._push_undo()
        if item.checkState() == Qt.Checked:
            self._excluded.discard(i)
        else:
            self._excluded.add(i)
        self._refresh_segments()

    def export(self):
        if not self.audio_path or not self.boundaries:
            QMessageBox.warning(self, "提示", "请先加载音频并标记至少一个边界。")
            return
        self._refresh_segments()
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not out_dir:
            return
        try:
            start = int(self.id_edit.text().strip() or "1")
        except ValueError:
            start = 1
        # 后台线程导出，界面保持响应。
        # 导出期间禁止 itemChanged 信号（进度更新会改列表项，避免误触发勾选处理），
        # 快照当前片段列表用于进度定位。
        self._export_snapshot = [self.seg_list.item(i) for i in range(self.seg_list.count())]
        self.seg_list.blockSignals(True)
        self.btn_export.setEnabled(False)
        self.statusBar().showMessage("正在后台导出…")
        self._export_worker = ExportWorker(
            self.audio_path, self._seg_times(), set(self._excluded),
            self.text_segments, out_dir, start)
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_progress(self, idx, seg_id, state):
        # 使用导出开始时的快照项；列表在导出中被 blockSignals 保护
        items = getattr(self, "_export_snapshot", [])
        if idx >= len(items):
            return
        item = items[idx]
        base = item.data(Qt.UserRole + 2)
        if base is None:
            base = item.data(Qt.UserRole)  # (s, e, i) 不含文本，回退用纯编号段
            base = f"{seg_id}  {self._fmt(base[0])} ~ {self._fmt(base[1])}"
        if state == "run":
            item.setText(f"{base}  ⟳ 转码中…")
        elif state == "ok":
            item.setText(f"{base}  ✓")
            item.setForeground(Qt.darkGreen)
        elif state == "skip":
            item.setText(f"{base}  — 跳过")
        elif state == "fail":
            item.setText(f"{base}  ✗ 失败")
            item.setForeground(Qt.red)

    def _on_export_done(self, ok, skipped, out_dir, errors):
        self.seg_list.blockSignals(False)
        self.btn_export.setEnabled(True)
        self._refresh_segments()  # 导出完成后重建干净列表（去进度标记）
        if errors:
            QMessageBox.warning(self, "完成(有失败)",
                                f"导出 {ok} 个，跳过 {skipped} 段。\n失败:\n" + "\n".join(errors[:5]))
        else:
            QMessageBox.information(
                self, "完成",
                f"导出 {ok} 个 MP3，跳过 {skipped} 段（不占编号）。\n{out_dir}\n并生成 对应表.md")
        self.statusBar().showMessage(f"导出完成: {out_dir}")

    # ---------- 项目 ----------
    def save_project(self):
        if not self.audio_path:
            return
        f, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "第1集裁切.json", "项目 (*.json)")
        if not f:
            return
        Path(f).write_text(json.dumps({
            "audio": self.audio_path,
            "text_file": getattr(self, "_text_path", None),
            "boundaries": self.boundaries,
            "start_id": self.id_edit.text(),
            "excluded": sorted(self._excluded),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        self.statusBar().showMessage(f"项目已保存: {f}")

    def open_project(self):
        f, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "项目 (*.json)")
        if not f:
            return
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        self.project_path = f
        self.id_edit.setText(d.get("start_id", "00201"))
        if d.get("audio") and Path(d["audio"]).is_file():
            self.load_audio(d["audio"])
        if d.get("text_file") and Path(d["text_file"]).is_file():
            self.load_text(Path(d["text_file"]))
        self.boundaries = d.get("boundaries", [])
        self._excluded = set(d.get("excluded", []))
        self._refresh_segments()

    @staticmethod
    def _fmt(t):
        t = max(0, t)
        return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
