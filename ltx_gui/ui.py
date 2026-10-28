"""Independent Qt frontend with manual spatial-mask selection and streaming processing."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPoint, QRect, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QDoubleSpinBox, QSpinBox,
    QTableView, QVBoxLayout, QWidget,
)

from .config import load_settings, save_settings
from .core import Job, JobQueue, Options, Rectangle, VIDEO_EXTENSIONS, plan_queue
from .runner import BatchRunner, Update


class QueueModel(QAbstractTableModel):
    columns = ("파일", "영역 / 마스크", "상태")

    def __init__(self, queue: JobQueue):
        super().__init__()
        self.queue = queue
        self.states: dict[int, str] = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.queue)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            item = self.queue[index.row()]
            region = item.rectangle.as_text() if item.rectangle else (Path(item.mask_path).name if item.mask_path else "미지정")
            return (Path(item.input_path).name, region, self.states.get(index.row(), "대기"))[index.column()]
        return None

    def add(self, jobs: list[Job]):
        self.beginResetModel()
        for job in jobs:
            self.queue.add(job)
        self.endResetModel()

    def refresh(self):
        self.beginResetModel()
        self.endResetModel()

    def remove(self, rows: list[int]):
        for row in sorted(rows, reverse=True):
            self.beginRemoveRows(QModelIndex(), row, row)
            self.queue.remove_at(row)
            self.endRemoveRows()
        self.states.clear()
        self.refresh()

    def set_state(self, index: int, state: str):
        self.states[index] = state
        self.dataChanged.emit(self.index(index, 2), self.index(index, 2))


class SelectionCanvas(QLabel):
    def __init__(self, frame):
        super().__init__()
        h, w = frame.shape[:2]
        scale = min(880/w, 520/h, 1.0)
        size = (max(1, round(w*scale)), max(1, round(h*scale)))
        preview = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
        ph, pw = preview.shape[:2]
        image = QImage(preview.data, pw, ph, 3*pw, QImage.Format.Format_RGB888).copy()
        self.pix = QPixmap.fromImage(image)
        self.setFixedSize(self.pix.size())
        self.start: QPoint | None = None
        self.end: QPoint | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.position().toPoint()
            self.end = self.start
            self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is not None:
            self.end = event.position().toPoint()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)
        if self.start and self.end:
            painter.setPen(QPen(QColor("#00d4ff"), 2))
            painter.fillRect(QRect(self.start, self.end).normalized(), QColor(0, 200, 255, 65))
            painter.drawRect(QRect(self.start, self.end).normalized())
        painter.end()

    def rectangle(self) -> Rectangle:
        if self.start is None or self.end is None:
            raise ValueError("복원할 영역을 마우스로 드래그하세요.")
        rect = QRect(self.start, self.end).normalized()
        w, h = self.width(), self.height()
        x0 = max(0, min(w, rect.left()))/w
        y0 = max(0, min(h, rect.top()))/h
        x1 = max(0, min(w, rect.right()+1))/w
        y1 = max(0, min(h, rect.bottom()+1))/h
        region = Rectangle(x0, y0, x1, y1)
        region.validate()
        return region


class RegionDialog(QDialog):
    def __init__(self, filename: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("첫 프레임에서 마스크 영역 지정")
        stream = cv2.VideoCapture(filename)
        try:
            okay, frame = stream.read()
        finally:
            stream.release()
        if not okay:
            raise ValueError("첫 번째 프레임을 열 수 없습니다.")
        self.canvas = SelectionCanvas(frame)
        panel = QVBoxLayout(self)
        panel.addWidget(QLabel("마우스로 복원 영역을 드래그하세요. 선택 영역만 변경됩니다."))
        panel.addWidget(self.canvas)
        row = QHBoxLayout()
        confirm = QPushButton("영역 적용")
        cancel = QPushButton("취소")
        confirm.clicked.connect(self._accept)
        cancel.clicked.connect(self.reject)
        row.addWidget(confirm)
        row.addWidget(cancel)
        panel.addLayout(row)
        self.selected: Rectangle | None = None

    def _accept(self):
        try:
            self.selected = self.canvas.rectangle()
        except ValueError as exc:
            QMessageBox.warning(self, "영역 오류", str(exc))
            return
        self.accept()


class Worker(QObject):
    update = Signal(object)
    finished = Signal()

    def __init__(self, planned, options):
        super().__init__()
        self.runner = BatchRunner(self.update.emit, options)
        self.planned = planned

    @Slot()
    def execute(self):
        try:
            self.runner.run(self.planned)
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LTX GUI | Independent Video Restoration")
        self.resize(1120, 850)
        try:
            opts, saved = load_settings()
        except (ValueError, OSError, KeyError) as exc:
            opts, saved = Options(), []
        self.queue = JobQueue(saved)
        self.model = QueueModel(self.queue)
        self.busy = False
        self.worker = None
        self.thread = None
        base = QWidget()
        self.setCentralWidget(base)
        layout = QVBoxLayout(base)

        description = QLabel("독립 영상 인페인팅 엔진 · Jasna 사용 안 함 · 지정한 영역만 복원")
        description.setStyleSheet("font-weight: 650; font-size: 17px; color: #247fc2;")
        layout.addWidget(description)
        row = QHBoxLayout()
        for title, call in (("영상 추가", self.add_files), ("폴더 추가", self.add_folder),
                            ("영역 선택", self.select_region), ("마스크 이미지", self.select_mask),
                            ("선택 제거", self.remove_selected)):
            button = QPushButton(title)
            button.clicked.connect(call)
            row.addWidget(button)
        layout.addLayout(row)

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        controls = QGridLayout()
        self.output = QLineEdit(opts.output_dir)
        self.output.setPlaceholderText("공백이면 원본 폴더")
        out_btn = QPushButton("찾기")
        out_btn.clicked.connect(self.choose_output)
        controls.addWidget(QLabel("출력 폴더"), 0, 0)
        controls.addWidget(self.output, 0, 1)
        controls.addWidget(out_btn, 0, 2)

        self.ffmpeg = QLineEdit(opts.ffmpeg)
        controls.addWidget(QLabel("FFmpeg 명령"), 1, 0)
        controls.addWidget(self.ffmpeg, 1, 1, 1, 2)

        self.method = QComboBox()
        self.method.addItem("빠른 공간 인페인팅 (Telea)", "fast")
        self.method.addItem("시간축 Optical Flow + Telea", "temporal")
        self.method.setCurrentIndex(0 if opts.method == "fast" else 1)
        self.encoder = QComboBox()
        self.encoder.addItem("CPU H.264 (libx264)", "libx264")
        self.encoder.addItem("NVIDIA GPU 인코딩 (NVENC)", "h264_nvenc")
        self.encoder.setCurrentIndex(0 if opts.encoder == "libx264" else 1)
        controls.addWidget(QLabel("복원 알고리즘"), 2, 0)
        controls.addWidget(self.method, 2, 1, 1, 2)
        controls.addWidget(QLabel("출력 인코딩"), 3, 0)
        controls.addWidget(self.encoder, 3, 1, 1, 2)

        self.radius = QSpinBox()
        self.radius.setRange(1, 20)
        self.radius.setValue(opts.radius)
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0, 0.8)
        self.weight.setSingleStep(0.05)
        self.weight.setValue(opts.temporal_weight)
        self.flow = QDoubleSpinBox()
        self.flow.setRange(0.125, 1.0)
        self.flow.setSingleStep(0.125)
        self.flow.setValue(opts.flow_scale)
        advanced = QHBoxLayout()
        for title, widget in (("공간 반경", self.radius), ("Temporal 가중치", self.weight), ("Flow 배율", self.flow)):
            advanced.addWidget(QLabel(title))
            advanced.addWidget(widget)
        controls.addLayout(advanced, 4, 0, 1, 3)
        layout.addLayout(controls)

        action = QHBoxLayout()
        self.start = QPushButton("▶ 복원 실행")
        self.start.clicked.connect(self.start_jobs)
        self.cancel = QPushButton("■ 안전 중지")
        self.cancel.clicked.connect(self.stop_jobs)
        self.cancel.setEnabled(False)
        save_btn = QPushButton("대기열·설정 저장")
        save_btn.clicked.connect(self.persist)
        for b in (self.start, self.cancel, save_btn):
            action.addWidget(b)
        layout.addLayout(action)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(1500)
        layout.addWidget(self.logs)
        self.statusBar().showMessage("독립 CPU 복원 파이프라인. GPU는 NVENC 인코딩에서만 사용합니다.")

    def selected_row(self):
        idx = self.table.selectionModel().selectedRows()
        if len(idx) != 1:
            raise ValueError("대기열에서 영상 하나를 선택하세요.")
        return idx[0].row()

    def add_files(self):
        if self.busy:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "영상 선택", "", "Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v)")
        self.model.add([Job(path) for path in paths])

    def add_folder(self):
        if self.busy:
            return
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            self.model.add([Job(str(p)) for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])

    def remove_selected(self):
        if not self.busy:
            self.model.remove([x.row() for x in self.table.selectionModel().selectedRows()])

    def select_region(self):
        if self.busy:
            return
        try:
            index = self.selected_row()
            dialog = RegionDialog(self.queue[index].input_path, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.queue.update_at(index, rectangle=dialog.selected, mask_path="")
                self.model.refresh()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "영역 선택", str(exc))

    def select_mask(self):
        if self.busy:
            return
        try:
            index = self.selected_row()
        except ValueError as exc:
            QMessageBox.warning(self, "마스크", str(exc))
            return
        path, _ = QFileDialog.getOpenFileName(self, "흰색=복원, 검정=보존", "", "Image (*.png *.bmp *.jpg *.jpeg)")
        if path:
            self.queue.update_at(index, rectangle=None, mask_path=path)
            self.model.refresh()

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "출력 폴더")
        if folder:
            self.output.setText(folder)

    def options(self):
        return Options(output_dir=self.output.text().strip(), method=self.method.currentData(),
                       encoder=self.encoder.currentData(), radius=self.radius.value(),
                       temporal_weight=self.weight.value(), flow_scale=self.flow.value(),
                       ffmpeg=self.ffmpeg.text().strip())

    def persist(self):
        try:
            save_settings(self.options(), self.queue.items())
            self.logs.appendPlainText("설정 저장 완료")
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "설정 저장", str(exc))

    def start_jobs(self):
        if self.busy or not len(self.queue):
            return
        try:
            opts = self.options()
            planned = plan_queue(self.queue, opts)
            save_settings(opts, self.queue.items())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "실행 전 검증", str(exc))
            return
        self.busy = True
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.table.setEnabled(False)
        self.progress.setValue(0)
        self.model.states.clear()
        self.model.refresh()
        self.thread = QThread(self)
        self.worker = Worker(planned, opts)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.execute)
        self.worker.update.connect(self.update_status)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.run_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(object)
    def update_status(self, update: Update):
        if update.kind in ("started", "completed", "failed", "stopped") and update.index >= 0:
            self.model.set_state(update.index, {"started":"실행 중", "completed":"완료", "failed":"실패", "stopped":"중지"}[update.kind])
        if update.kind == "progress":
            if update.percent >= 0:
                self.progress.setValue(max(0, min(100, round(update.percent))))
            self.statusBar().showMessage(f"복원 처리 {update.fps:.1f} fps | {update.message}")
        elif update.kind != "progress":
            self.logs.appendPlainText(f"[{update.kind}] {update.message}")

    @Slot()
    def run_finished(self):
        self.busy = False
        self.start.setEnabled(True)
        self.cancel.setEnabled(False)
        self.table.setEnabled(True)
        self.worker = None
        self.thread = None

    def stop_jobs(self):
        if self.worker:
            self.cancel.setEnabled(False)
            self.worker.runner.request_stop()

    def closeEvent(self, event):
        if self.busy:
            QMessageBox.information(self, "처리 중", "작업을 중지한 후 종료하세요.")
            event.ignore()
            return
        try:
            save_settings(self.options(), self.queue.items())
        except (ValueError, OSError):
            pass
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LTX GUI — Independent")
    window = MainWindow()
    window.show()
    return app.exec()
