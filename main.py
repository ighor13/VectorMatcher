import sys
import os
import traceback
import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableView,
    QPushButton, QLineEdit, QLabel, QPlainTextEdit, QFileDialog,
    QMenu, QSplitter, QProgressBar, QHeaderView, QSizePolicy
)

from worker import Matcher

# --- НАСТРОЙКА ПУТЕЙ ---
if getattr(sys, 'frozen', False):
    # Если exe, ищем файлы рядом с exe
    base_dir = os.path.dirname(sys.executable)
else:
    # Если скрипт
    base_dir = os.path.dirname(os.path.abspath(__file__))

# --- ПЕРЕХВАТЧИК КОНСОЛИ (STDOUT/STDERR) ---
class StreamRedirector:
    def __init__(self, signal):
        self.signal = signal
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer or '\r' in self.buffer:
            if '\n' in self.buffer:
                idx = self.buffer.index('\n')
            else:
                idx = self.buffer.index('\r')
            
            line = self.buffer[:idx].strip()
            self.buffer = self.buffer[idx+1:]
            
            if line:
                self.signal.emit(line)

    def flush(self):
        pass


class PandasModel(QAbstractTableModel):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else None
        self._headers = list(self._df.columns) if self._df is not None else []
        self._yellow = 0.8
        self._red = 0.7

    def set_df(self, df):
        self.beginResetModel()
        self._df = df
        self._headers = list(df.columns) if df is not None else []
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=None):
        return 0 if self._df is None else len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return str(self._df.iat[index.row(), index.column()])
        if role == Qt.ForegroundRole and self._headers[index.column()] == "score":
            try:
                s = float(self._df.iat[index.row(), index.column()])
            except Exception:
                return None
            if s >= self._yellow:
                return QColor("green")
            if s >= self._red:
                return QColor("orange")
            return QColor("red")
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return super().headerData(section, orientation, role)


class WorkerThread(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, old_df, new_df, model_path):
        super().__init__()
        self.old_df = old_df
        self.new_df = new_df
        self.model_path = model_path

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirector = StreamRedirector(self.log)
        sys.stdout = redirector
        sys.stderr = redirector

        try:
            self.log.emit("Инициализация модели...")
            matcher = Matcher(self.model_path, use_gpu=True)
            self.log.emit("Модель загружена в память.")
            
            df = matcher.run(
                self.old_df, self.new_df,
                progress_cb=self.progress.emit,
                log_cb=self.log.emit,
            )
            self.done.emit(df)
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print("!!! ОШИБКА В ПОТОКЕ !!!")
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class MatcherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vector Matcher")
        self.resize(1400, 900)

        self.model_old = PandasModel()
        self.model_new = PandasModel()
        self.model_out = PandasModel()

        self.view_old = self._make_view(self.model_old, loadable=True, savable=False)
        self.view_new = self._make_view(self.model_new, loadable=True, savable=False)
        self.view_out = self._make_view(self.model_out, loadable=False, savable=True, stretch_last=True)

        self.btn_go = QPushButton("GO")
        self.btn_go.clicked.connect(self.start_work)
        self.btn_go.setFixedWidth(60)

        self.edit_yellow = QLineEdit("0.80")
        self.edit_yellow.setFixedWidth(50)
        self.edit_red = QLineEdit("0.70")
        self.edit_red.setFixedWidth(50)

        self.btn_model = QPushButton("BGE-M3 (по умолчанию)")
        self.btn_model.clicked.connect(self.pick_model)
        self.model_path = "BAAI/bge-m3"

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setTextVisible(True)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_go)
        top_bar.addWidget(QLabel("жёлтый:"))
        top_bar.addWidget(self.edit_yellow)
        top_bar.addWidget(QLabel("красный:"))
        top_bar.addWidget(self.edit_red)
        top_bar.addWidget(self.btn_model)
        top_bar.addWidget(self.progress_bar)
        top_bar.addStretch()

        right_panel = QVBoxLayout()
        right_panel.addLayout(top_bar)
        right_panel.addWidget(self.view_out)

        top_split = QHBoxLayout()
        top_split.addWidget(self.view_old, 1)
        top_split.addWidget(self.view_new, 1)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        top_split.addWidget(right_widget, 1)

        top_container = QWidget()
        top_container.setLayout(top_split)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_container)
        splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)

        main = QVBoxLayout(self)
        main.addWidget(splitter)

        self.thread = None
        self._load_initial_data()

    def _log(self, msg):
        self.log.appendPlainText(msg)

    def _load_initial_data(self):
        old_path = os.path.join(base_dir, "old.tsv")
        new_path = os.path.join(base_dir, "new.tsv")

        try:
            old_df = pd.read_csv(old_path, sep="\t", header=None, names=["id", "data"])
            old_df["id"] = old_df["id"].fillna("").astype(str)
            old_df["data"] = old_df["data"].fillna("").astype(str).str.strip()
            self.model_old.set_df(old_df)
            self._log(f"Тест old загружен: {len(old_df)} строк")
            self._apply_initial_widths(self.view_old, self.model_old)
        except Exception as e:
            self._log(f"Не удалось загрузить old.tsv: {e}")

        try:
            new_df = pd.read_csv(new_path, sep="\t", header=None, names=["id", "data"])
            new_df["id"] = new_df["id"].fillna("").astype(str)
            new_df["data"] = new_df["data"].fillna("").astype(str).str.strip()
            self.model_new.set_df(new_df)
            self._log(f"Тест new загружен: {len(new_df)} строк")
            self._apply_initial_widths(self.view_new, self.model_new)
        except Exception as e:
            self._log(f"Не удалось загрузить new.tsv: {e}")

    def _apply_initial_widths(self, view, model):
        for i, name in enumerate(model._headers):
            if name == "id":
                view.setColumnWidth(i, 100)
            elif name == "data":
                view.setColumnWidth(i, 300)

    def _make_view(self, model, loadable, savable, stretch_last=True):
        view = QTableView()
        view.setModel(model)
        view.setContextMenuPolicy(Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(
            lambda pos, v=view: self.show_menu(v, pos, loadable, savable)
        )
        
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view.setWordWrap(True)
        
        header = view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(stretch_last) 
        
        view.verticalHeader().setDefaultSectionSize(60)
        
        if loadable:
            view.setAcceptDrops(True)
            view.dragEnterEvent = lambda e, v=view: self._drag_enter(e, v)
            view.dropEvent = lambda e, v=view: self._drop(e, v)
        return view

    def _drag_enter(self, event, view):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drop(self, event, view):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.load_into_view(view, path)
                break

    def load_into_view(self, view, path):
        try:
            df = pd.read_csv(path, sep="\t", header=None, names=["id", "data"])
            df["id"] = df["id"].fillna("").astype(str)
            df["data"] = df["data"].fillna("").astype(str).str.strip()
            model = view.model()
            if isinstance(model, PandasModel):
                model.set_df(df)
                self._apply_initial_widths(view, model)
            self._log(f"Загружено {len(df)} строк из {path}")
        except Exception as e:
            self._log(f"Ошибка загрузки {path}: {e}")

    def show_menu(self, view, pos, loadable, savable):
        menu = QMenu(view)
        if loadable:
            act = menu.addAction("Load file...")
            act.triggered.connect(lambda: self.load_dialog(view))
        if savable:
            act = menu.addAction("Save as...")
            act.triggered.connect(self.save_out)
        if not menu.isEmpty():
            menu.exec(view.viewport().mapToGlobal(pos))

    def load_dialog(self, view):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть", "", "TSV (*.tsv *.txt *.csv);;All (*)")
        if path:
            self.load_into_view(view, path)

    def save_out(self):
        df = self.model_out._df
        if df is None or df.empty:
            self._log("Нет результата для сохранения.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить как", "", "TSV (*.tsv *.txt)")
        if not path:
            return
        df.to_csv(path, sep="\t", index=False, header=False)
        self._log(f"Сохранено в {path}")

    def pick_model(self):
        path = QFileDialog.getExistingDirectory(self, "Выбрать папку с моделью")
        if path:
            self.model_path = path
            self.btn_model.setText(path)
            self._log(f"Выбрана модель: {path}")

    def start_work(self):
        if self.thread and self.thread.isRunning():
            return
        old_df = self.model_old._df
        new_df = self.model_new._df
        if old_df is None or new_df is None or old_df.empty or new_df.empty:
            self._log("Загрузи old и new файлы.")
            return

        try:
            self.model_out._yellow = float(self.edit_yellow.text())
            self.model_out._red = float(self.edit_red.text())
        except ValueError:
            self._log("Пороги должны быть числами.")
            return

        self.btn_go.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.thread = WorkerThread(old_df, new_df, self.model_path)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log.connect(self._log)
        self.thread.done.connect(self.on_done)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_done(self, df):
        self.model_out.set_df(df)
        
        for i, name in enumerate(self.model_out._headers):
            if name == "old_id":
                self.view_out.setColumnWidth(i, 50)  
            elif name == "old_data":
                self.view_out.setColumnWidth(i, 170)
            elif name == "new_id":
                self.view_out.setColumnWidth(i, 50)  
            elif name == "new_data":
                self.view_out.setColumnWidth(i, 170)
            elif name == "score":
                self.view_out.setColumnWidth(i, 70) 
                
        self.btn_go.setEnabled(True)
        self.progress_bar.setValue(100)
        self._log(f"Готово: {len(df)} строк результата.")

    def on_error(self, msg):
        self._log(f"Ошибка: {msg}")
        self.btn_go.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    w = MatcherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()