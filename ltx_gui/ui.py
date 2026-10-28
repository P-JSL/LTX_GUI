"""PySide6 Windows/Linux GUI; compute engine stays in an external Jasna process."""
from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTableView, QVBoxLayout, QWidget,
)

from .config import load_settings, save_settings
from .core import Job, JobQueue, Options, QUALITY_MODELS, VIDEO_EXTENSIONS, plan_queue, validate_segments
from .runner import BatchRunner, Update
from .telemetry import cpu_ram_stats, gpu_stats


class QueueTable(QAbstractTableModel):
    HEADERS = ("파일", "복원 구간", "상태", "출력")

    def __init__(self, queue: JobQueue) -> None:
        super().__init__()
        self.queue = queue
        self.states: dict[int, str] = {}

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.queue)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 4

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            job = self.queue[index.row()]
            return (Path(job.input_path).name, job.segments or "전체", self.states.get(index.row(), "대기"),
                    Path(job.output_path).name if job.output_path else "자동 생성")[index.column()]
        if role == Qt.ItemDataRole.ForegroundRole:
            status = self.states.get(index.row(), "")
            if status == "완료":
                return QColor("#66d5a4")
            if status in ("실패", "중지"):
                return QColor("#ff8a8a")
        return None

    def add_jobs(self, jobs: list[Job]) -> int:
        valid = []
        seen: set[str] = set()
        for job in jobs:
            path = Path(job.input_path)
            key = os.path.normcase(str(path.expanduser().resolve()))
            if key not in seen and not self.queue.contains(job.input_path):
                seen.add(key)
                valid.append(job)
        if not valid:
            return 0
        first = len(self.queue)
        self.beginInsertRows(QModelIndex(), first, first + len(valid) - 1)
        for job in valid:
            self.queue.add(job)
        self.endInsertRows()
        return len(valid)

    def remove_rows(self, indices: list[int]):
        for row in sorted(set(indices), reverse=True):
            if 0 <= row < len(self.queue):
                self.beginRemoveRows(QModelIndex(), row, row)
                self.queue.remove_at(row)
                self.endRemoveRows()
        self.states.clear()

    def clear(self) -> None:
        self.beginResetModel()
        self.queue.clear()
        self.states.clear()
        self.endResetModel()

    def edit_segments(self, row: int, segments: str) -> None:
        # Replacing a selected row preserves its place in the queue.
        self.queue.update_at(row, segments=segments)
        self.dataChanged.emit(self.index(row, 1), self.index(row, 1))

    def set_state(self, row: int, state: str) -> None:
        self.states[row] = state
        self.dataChanged.emit(self.index(row, 2), self.index(row, 2))

    def reset_statuses(self) -> None:
        self.states.clear()
        if len(self.queue):
            self.dataChanged.emit(self.index(0, 2), self.index(len(self.queue)-1, 2))


class DropTable(QTableView):
    file_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QTableView.DragDropMode.DropOnly)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            self.file_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class RunWorker(QObject):
    update = Signal(object)
    finished = Signal()

    def __init__(self, planned) -> None:
        super().__init__()
        self.planned = planned
        self.runner = BatchRunner(self.update.emit)

    @Slot()
    def execute(self) -> None:
        try:
            self.runner.run(self.planned)
        except Exception as exc:
            self.update.emit(Update(-1, "failed", f"작업자 예외: {exc}"))
        finally:
            self.finished.emit()


class ResourceWorker(QThread):
    stats = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            cpu, ram = cpu_ram_stats()
            devices = gpu_stats()
            parts = [f"CPU {cpu:.0f}%" if cpu is not None else "CPU -",
                     f"RAM {ram:.0f}%" if ram is not None else "RAM -"]
            if devices:
                parts.extend(f"GPU {d.index} {d.utilization_percent}% / VRAM {d.used_mib:,}/{d.total_mib:,} MiB"
                             for d in devices)
            else:
                parts.append("NVIDIA GPU 정보 없음 (AMD/기타 GPU는 실행 엔진에서 확인)")
            self.stats.emit("   |   ".join(parts))
            self._halt.wait(3)

    def stop(self) -> None:
        self._halt.set()


STYLE = """
QWidget { font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; font-size: 12px;
          background: #111823; color: #e5ecf3; }
QMainWindow { background: #0e1520; }
QFrame#TopBanner { background: #192335; border-radius: 12px; }
QLabel#AppTitle { font-size: 24px; font-weight: 750; color: #f2f7ff; }
QLabel#Muted { color: #96a6b8; }
QGroupBox { border: 1px solid #2c3a4c; border-radius: 9px;
            margin-top: 12px; padding-top: 14px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 7px; }
QPushButton { background: #25344b; padding: 8px 14px;
              border: 1px solid #354960; border-radius: 7px; }
QPushButton:hover { background: #344967; }
QPushButton:disabled { background: #192230; color: #65758a; }
QPushButton#Primary { background: #356ee7; border-color: #477ef1; font-weight: 700; }
QPushButton#Primary:hover { background: #4c82f3; }
QPushButton#Danger { background: #56313e; border-color: #814352; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit { background: #172334;
    color: #eaf1ff; border: 1px solid #37475c; border-radius: 5px; padding: 6px; }
QComboBox QAbstractItemView { background: #202e41; }
QTableView { background: #152131; alternate-background-color: #19283a;
             gridline-color: #2c3a50; selection-background-color: #285390;
             border: 1px solid #304156; border-radius: 7px; }
QHeaderView::section { background: #202f44; color: #c9d6e6; padding: 8px;
                        border: none; border-right: 1px solid #304054; }
QProgressBar { background: #202c3a; color: white; border: 1px solid #344658;
               border-radius: 5px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #3675e3; border-radius: 4px; }
QScrollArea { border: 0; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LTX GUI  |  영상 복원 작업 관리자")
        self.resize(1340, 860)
        self.setMinimumSize(990, 660)
        options, stored_jobs = load_settings()
        self.queue = JobQueue(stored_jobs)
        self.model = QueueTable(self.queue)
        self._thread: QThread | None = None
        self._worker: RunWorker | None = None
        self._planned = []
        self._busy = False
        self._build(options)
        self.resources = ResourceWorker(self)
        self.resources.stats.connect(self.stats_label.setText)
        self.resources.start()
        self._log("프로그램 준비 완료. Jasna LTX 모델은 별도로 설치 및 정품 활성화해야 합니다.")

    def _build(self, options: Options) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 15, 18, 15)
        banner = QFrame(objectName="TopBanner")
        top = QVBoxLayout(banner)
        title = QLabel("LTX GUI", objectName="AppTitle")
        top.addWidget(title)
        top.addWidget(QLabel("고성능 영상 복원 · GPU 작업 대기열 · LTX / BasicVSR++ 실행 제어", objectName="Muted"))
        layout.addWidget(banner)

        main = QHBoxLayout()
        main.setSpacing(13)
        layout.addLayout(main, stretch=1)
        left = QVBoxLayout()
        main.addLayout(left, stretch=3)
        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setMinimumWidth(345)
        right.setMaximumWidth(450)
        main.addWidget(right, stretch=2)
        settings_holder = QWidget()
        right.setWidget(settings_holder)
        settings_layout = QVBoxLayout(settings_holder)
        settings_layout.setContentsMargins(2, 2, 2, 2)

        queue_group = QGroupBox("작업 대기열  ·  파일을 여기로 끌어놓을 수 있습니다")
        queue_box = QVBoxLayout(queue_group)
        self.table = DropTable()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.file_dropped.connect(self._add_paths)
        queue_box.addWidget(self.table)
        btns = QHBoxLayout()
        for text, handler in (("파일 추가", self._add_files), ("폴더 추가", self._add_folder),
                              ("선택 제거", self._remove_selected), ("전체 지우기", self._clear_jobs),
                              ("영상 열기", self._open_input)):
            button = QPushButton(text)
            button.clicked.connect(handler)
            btns.addWidget(button)
        queue_box.addLayout(btns)
        left.addWidget(queue_group, stretch=4)

        segment_group = QGroupBox("선택 영상의 복원 범위")
        seg_box = QVBoxLayout(segment_group)
        seg_box.addWidget(QLabel("비워 두면 전체 영상 / 예: 00:10-00:25,01:10-01:30"))
        segrow = QHBoxLayout()
        self.segments = QLineEdit()
        self.segments.setPlaceholderText("시작-종료,시작-종료")
        segrow.addWidget(self.segments, 1)
        self.apply_segments = QPushButton("범위 적용")
        self.apply_segments.clicked.connect(self._set_segments)
        segrow.addWidget(self.apply_segments)
        seg_box.addLayout(segrow)
        left.addWidget(segment_group)
        self.table.selectionModel().selectionChanged.connect(self._select_row)

        info_group = QGroupBox("진행 상태 / 실행 로그")
        info_box = QVBoxLayout(info_group)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        info_box.addWidget(self.progress)
        self.stage_label = QLabel("대기")
        info_box.addWidget(self.stage_label)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.document().setMaximumBlockCount(1500)
        info_box.addWidget(self.logs, stretch=1)
        left.addWidget(info_group, stretch=2)

        paths_group = QGroupBox("실행 환경")
        path_form = QFormLayout(paths_group)
        self.exec_edit = QLineEdit(options.executable)
        exe_row = QHBoxLayout()
        exe_row.addWidget(self.exec_edit, 1)
        browse_exe = QPushButton("찾기")
        browse_exe.clicked.connect(self._browse_executable)
        exe_row.addWidget(browse_exe)
        path_form.addRow("Jasna 실행 파일", exe_row)
        self.output_edit = QLineEdit(options.output_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit, 1)
        browse_out = QPushButton("찾기")
        browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(browse_out)
        path_form.addRow("출력 폴더", out_row)
        settings_layout.addWidget(paths_group)

        model_group = QGroupBox("복원 모델")
        model_form = QFormLayout(model_group)
        self.quality = QComboBox()
        self.quality.addItems(list(QUALITY_MODELS))
        self.quality.setCurrentText(options.quality)
        model_form.addRow("품질", self.quality)
        self.gpu = QSpinBox()
        self.gpu.setRange(0, 63)
        self.gpu.setValue(options.gpu_index)
        model_form.addRow("GPU 번호", self.gpu)
        self.seed = QLineEdit(str(options.seed))
        model_form.addRow("시드", self.seed)
        self.fast = QCheckBox("RTX 50 전용 빠른 모드")
        self.fast.setChecked(options.fast)
        model_form.addRow(self.fast)
        self.large = QCheckBox("큰 모자이크 고품질 처리 (VRAM 증가)")
        self.large.setChecked(options.large_canvas)
        model_form.addRow(self.large)
        self.overlap = QCheckBox("LTX 프레임 겹침 최소화")
        self.overlap.setChecked(options.less_overlap)
        model_form.addRow(self.overlap)
        self.trial = QCheckBox("LTX 속도 테스트 (영상 복원 안 함)")
        self.trial.setChecked(options.trial)
        model_form.addRow(self.trial)
        self.auto_split = QSpinBox()
        self.auto_split.setRange(0, 100)
        self.auto_split.setSuffix(" %  (0=사용 안 함)")
        self.auto_split.setValue(options.auto_split_percent)
        model_form.addRow("LTX 큰 영역 우선", self.auto_split)
        settings_layout.addWidget(model_group)

        encode_group = QGroupBox("인코딩 / BasicVSR++ 고급 설정")
        enc_form = QFormLayout(encode_group)
        self.codec = QComboBox()
        self.codec.addItems(["hevc", "h264", "av1"])
        self.codec.setCurrentText(options.codec)
        enc_form.addRow("영상 코덱", self.codec)
        self.cq = QSpinBox()
        self.cq.setRange(-1, 63)
        self.cq.setSpecialValueText("GPU 기본값")
        self.cq.setValue(-1 if options.cq is None else options.cq)
        enc_form.addRow("CQ (낮을수록 고화질)", self.cq)
        self.clip = QSpinBox()
        self.clip.setRange(2, 10000)
        self.clip.setValue(options.basic_clip_size)
        enc_form.addRow("BasicVSR++ 클립 프레임", self.clip)
        self.basic_overlap = QSpinBox()
        self.basic_overlap.setRange(0, 4999)
        self.basic_overlap.setValue(options.basic_overlap)
        enc_form.addRow("BasicVSR++ 프레임 겹침", self.basic_overlap)
        settings_layout.addWidget(encode_group)
        settings_layout.addStretch(1)

        toolbar = QHBoxLayout()
        self.start_button = QPushButton("▶  대기열 실행")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._start)
        toolbar.addWidget(self.start_button)
        self.stop_button = QPushButton("■  중지 / 체크포인트 저장")
        self.stop_button.setObjectName("Danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        toolbar.addWidget(self.stop_button)
        save_btn = QPushButton("설정 저장")
        save_btn.clicked.connect(self._save)
        toolbar.addWidget(save_btn)
        layout.addLayout(toolbar)
        self.stats_label = QLabel("자원 정보 확인 중...", objectName="Muted")
        layout.addWidget(self.stats_label)
        self.statusBar().showMessage("LTX GUI · Jasna 프로그램 및 라이선스는 별도로 설치해야 합니다.")
        self.setStyleSheet(STYLE)
        self.quality.currentTextChanged.connect(self._quality_changed)
        self._quality_changed(self.quality.currentText())

    def _quality_changed(self, quality: str) -> None:
        ltx = quality != "BasicVSR++"
        for control in (self.seed, self.fast, self.large, self.overlap, self.trial, self.auto_split):
            control.setEnabled(ltx and not self._busy)
        for control in (self.clip, self.basic_overlap):
            control.setEnabled(not ltx and not self._busy)

    def _options(self) -> Options:
        try:
            seed = int(self.seed.text())
        except ValueError:
            raise ValueError("시드는 정수로 입력하세요.") from None
        return Options(executable=self.exec_edit.text().strip(),
                       output_dir=self.output_edit.text().strip(),
                       quality=self.quality.currentText(), gpu_index=self.gpu.value(),
                       codec=self.codec.currentText(), seed=seed,
                       fast=self.fast.isChecked(), large_canvas=self.large.isChecked(),
                       less_overlap=self.overlap.isChecked(), trial=self.trial.isChecked(),
                       auto_split_percent=self.auto_split.value(),
                       basic_clip_size=self.clip.value(), basic_overlap=self.basic_overlap.value(),
                       cq=None if self.cq.value() == -1 else self.cq.value())

    def _log(self, message: str) -> None:
        self.logs.appendPlainText(message)

    def _browse_executable(self):
        result, _ = QFileDialog.getOpenFileName(self, "Jasna 실행 파일 선택", "", "실행 파일 (*.exe);;모든 파일 (*)")
        if result:
            self.exec_edit.setText(result)

    def _browse_output(self):
        result = QFileDialog.getExistingDirectory(self, "영상 출력 폴더 선택")
        if result:
            self.output_edit.setText(result)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "영상 파일 추가", "", "영상 (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts)")
        self._add_paths(paths)

    def _add_folder(self):
        path = QFileDialog.getExistingDirectory(self, "폴더에서 영상 추가")
        if path:
            self._add_paths([path])

    def _add_paths(self, paths: list[str]) -> None:
        if self._busy:
            return
        candidates = []
        for name in paths:
            path = Path(name)
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(Job(str(path.resolve())))
            elif path.is_dir():
                # One filesystem traversal, no O(n^2) duplicate comparisons.
                candidates.extend(Job(str(p.resolve())) for p in path.rglob("*")
                                  if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        count = self.model.add_jobs(candidates)
        self._log(f"대기열에 {count}개 파일을 추가했습니다. (중복은 무시)")

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def _select_row(self, *args) -> None:
        rows = self._selected_rows()
        if len(rows) == 1:
            self.segments.setText(self.queue[rows[0]].segments)
        else:
            self.segments.clear()

    def _set_segments(self):
        if self._busy:
            return
        rows = self._selected_rows()
        if len(rows) != 1:
            QMessageBox.information(self, "영상 선택", "영상을 1개 선택하세요.")
            return
        try:
            segments = validate_segments(self.segments.text())
        except ValueError as exc:
            QMessageBox.warning(self, "범위 오류", str(exc))
            return
        self.model.edit_segments(rows[0], segments)

    def _remove_selected(self):
        if not self._busy:
            self.model.remove_rows(self._selected_rows())

    def _clear_jobs(self):
        if not self._busy:
            self.model.clear()

    def _open_input(self):
        rows = self._selected_rows()
        if len(rows) == 1:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.queue[rows[0]].input_path))

    def _save(self):
        try:
            opts = self._options()
            save_settings(opts, self.queue.items())
            self._log("설정 및 대기열 저장 완료")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "저장 오류", str(exc))

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.apply_segments.setEnabled(not busy)
        self.table.setEnabled(not busy)
        self.quality.setEnabled(not busy)
        for control in (self.exec_edit, self.output_edit, self.codec, self.cq, self.gpu):
            control.setEnabled(not busy)
        self._quality_changed(self.quality.currentText())

    def _start(self):
        if self._busy or not len(self.queue):
            return
        try:
            opts = self._options()
            if opts.trial:
                answer = QMessageBox.question(self, "시험 실행", "LTX Trial은 복원을 수행하지 않습니다. 실행하시겠습니까?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
            executable = Path(opts.executable)
            if not executable.is_file() and shutil.which(opts.executable) is None:
                raise ValueError("Jasna 실행 파일을 찾지 못했습니다. 공식 배포본의 jasna.exe 경로를 선택하세요.")
            planned = plan_queue(self.queue, opts)
            save_settings(opts, self.queue.items())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "실행 전 검사 실패", str(exc))
            return
        self._planned = planned
        self.model.reset_statuses()
        self.progress.setValue(0)
        self._set_busy(True)
        thread = QThread(self)
        worker = RunWorker(planned)
        worker.moveToThread(thread)
        thread.started.connect(worker.execute)
        worker.update.connect(self._update)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._finished)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._log(f"대기열 실행: 총 {len(planned)}개 · GPU cuda:{opts.gpu_index}")
        thread.start()

    @Slot(object)
    def _update(self, event: Update):
        if event.kind == "all_done":
            self._log(event.message)
            return
        if event.index >= 0:
            status = {"started": "실행 중", "completed": "완료", "failed": "실패", "stopped": "중지"}.get(event.kind)
            if status:
                self.model.set_state(event.index, status)
            if event.kind == "started":
                self.progress.setValue(0)
            elif event.kind == "progress":
                self.progress.setValue(max(0, min(100, round(event.percent))))
            elif event.kind == "completed":
                self.progress.setValue(100)
            self.stage_label.setText(f"[{event.index + 1}/{len(self._planned)}] {status or event.message[:110]}")
        if event.kind in ("log", "failed", "completed", "stopped") and event.message:
            self._log(event.message)

    @Slot()
    def _finished(self):
        self._set_busy(False)
        self._thread = None
        self._worker = None
        self._log("작업 스레드 종료")

    def _stop(self):
        worker = self._worker
        if worker:
            self.stop_button.setEnabled(False)
            self._log("작업 중단을 요청했습니다. 체크포인트 저장을 위해 인터럽트를 우선 보냅니다.")
            worker.runner.request_stop()

    def closeEvent(self, event):
        if self._busy:
            QMessageBox.information(self, "작업 실행 중", "작업 중지 후 종료하세요. 진행 상태가 저장될 수 있습니다.")
            event.ignore()
            return
        self.resources.stop()
        self.resources.wait(3500)
        try:
            save_settings(self._options(), self.queue.items())
        except (OSError, ValueError):
            pass
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LTX GUI")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()
