import math

from typing import Optional
def rhoo(T):    # density
    return 3.20539 - 0.00962 * T + 9.55357e-6 * T **2 
def lambb(T):   # thermal conductive coefficent
    return 0.00727 + 5.62879e-5 * T + 2.23214e-8 * T**2
def nuu(T):     # dynamic viscosity
    return -1.13502e-6 + 8.823987e-8 * T - 5.60362e-11 * T**2


def Dhh(V, A):  # Hydraulic diameter
    return (32*1e-6-V)*4/A
def Ree(T, dh, v):     # Reynolds number
    return rhoo(T)*dh*v/nuu(T)
def Nuu(h, dh, T):     #  Nusselt number  
    return h*dh/lambb(T)

def h_10(t_out, t_in, tpms) :
    return (t_out - t_in) * 2 /(2 * tpms - t_in - t_out)
def h_20(t_out, t_in, tpms) :
    delta_T1 = tpms - t_in
    delta_T2 = tpms - t_out    
    return (t_out - t_in) /(tpms - (delta_T1 - delta_T2)/math.log(delta_T1 / delta_T2))
def h_1(rho, a0, v, cp, tpms, h10, a1):   # saunshu  
    return rho * v * a0 *cp * h10/a1
def h_2(rho, a0, v, cp, tpms, h20, a1):   # duishu 
    return rho * v * a0 *cp * h20/a1

def solve(v, tpms, t_out, V = 0, A = 1):
    a0 = 4e-4
    a1 = 16e-4
    t_in = 293.15
    cp = 1010
    T = (t_in + t_out)/2
    rho = rhoo(T)
    Dh = Dhh(V, A)
    h10 = h_10(t_out, t_in, tpms)
    h20 = h_20(t_out, t_in, tpms)
    h1 = h_1(rho, a0, v, cp, tpms, h10, a1)
    h2 = h_2(rho, a0, v, cp, tpms, h20, a1)
    Re = Ree(T, Dh, v)
    Nu1 =  Nuu(h1, Dh, T)
    Nu2 =  Nuu(h2, Dh, T)
    return h1, h2, Nu1, Nu2, Re


def _import_qt():
    try:
        from PyQt6 import QtCore, QtWidgets
        qt_api = "PyQt6"
    except Exception:
        try:
            from PySide6 import QtCore, QtWidgets
            qt_api = "PySide6"
        except Exception:
            from PyQt5 import QtCore, QtWidgets
            qt_api = "PyQt5"

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    except Exception:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

    return qt_api, QtCore, QtWidgets, FigureCanvasQTAgg, NavigationToolbar2QT


def run_gui() -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    qt_api, QtCore, QtWidgets, FigureCanvasQTAgg, NavigationToolbar2QT = _import_qt()

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"Nu-Re 交互绘图 ({qt_api})")
            self.resize(980, 720)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)

            form_row = QtWidgets.QHBoxLayout()
            root.addLayout(form_row)

            form = QtWidgets.QFormLayout()
            form_row.addLayout(form, 2)

            self.tpms = QtWidgets.QDoubleSpinBox()
            self.tpms.setRange(-1e9, 1e9)
            self.tpms.setDecimals(3)
            self.tpms.setValue(330.0)

            self.t_out = QtWidgets.QDoubleSpinBox()
            self.t_out.setRange(-1e9, 1e9)
            self.t_out.setDecimals(3)
            self.t_out.setValue(310.0)

            self.v_min = QtWidgets.QDoubleSpinBox()
            self.v_min.setRange(0.0, 1e9)
            self.v_min.setDecimals(6)
            self.v_min.setValue(0.2)

            self.v_max = QtWidgets.QDoubleSpinBox()
            self.v_max.setRange(0.0, 1e9)
            self.v_max.setDecimals(6)
            self.v_max.setValue(3.0)

            self.n_points = QtWidgets.QSpinBox()
            self.n_points.setRange(2, 5000)
            self.n_points.setValue(25)

            self.V = QtWidgets.QDoubleSpinBox()
            self.V.setRange(-1e9, 1e9)
            self.V.setDecimals(6)
            self.V.setValue(0.0)

            self.A = QtWidgets.QDoubleSpinBox()
            self.A.setRange(1e-12, 1e9)
            self.A.setDecimals(6)
            self.A.setValue(1.0)

            self.y_mode = QtWidgets.QComboBox()
            self.y_mode.addItems(["Nu1 & Nu2", "h1 & h2"])

            form.addRow("tpms", self.tpms)
            form.addRow("t_out", self.t_out)
            form.addRow("v_min", self.v_min)
            form.addRow("v_max", self.v_max)
            form.addRow("n_points", self.n_points)
            form.addRow("V", self.V)
            form.addRow("A", self.A)
            form.addRow("绘制内容", self.y_mode)

            btn_col = QtWidgets.QVBoxLayout()
            form_row.addLayout(btn_col, 1)
            self.btn_plot = QtWidgets.QPushButton("绘图")
            self.btn_plot.setMinimumHeight(36)
            btn_col.addWidget(self.btn_plot)
            btn_col.addStretch(1)

            self.fig = Figure(figsize=(7.5, 5.2), dpi=120)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            root.addWidget(self.toolbar)
            root.addWidget(self.canvas, 1)

            self.status = QtWidgets.QLabel("")
            root.addWidget(self.status)

            self.btn_plot.clicked.connect(self._plot)
            QtCore.QTimer.singleShot(0, self._plot)

        def _plot(self) -> None:
            tpms = float(self.tpms.value())
            t_out = float(self.t_out.value())
            v_min = float(self.v_min.value())
            v_max = float(self.v_max.value())
            n_points = int(self.n_points.value())
            V = float(self.V.value())
            A = float(self.A.value())

            if not (v_max > v_min):
                self.status.setText("v_max 必须大于 v_min")
                return

            vs = np.linspace(v_min, v_max, n_points)
            h1s = np.zeros_like(vs, dtype=float)
            h2s = np.zeros_like(vs, dtype=float)
            nu1s = np.zeros_like(vs, dtype=float)
            nu2s = np.zeros_like(vs, dtype=float)
            res = np.zeros_like(vs, dtype=float)

            for i, v in enumerate(vs.tolist()):
                h1, h2, Nu1, Nu2, Re = solve(v, tpms, t_out, V=V, A=A)
                h1s[i] = float(h1)
                h2s[i] = float(h2)
                nu1s[i] = float(Nu1)
                nu2s[i] = float(Nu2)
                res[i] = float(Re)

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            mode = str(self.y_mode.currentText())
            if mode == "h1 & h2":
                ax.plot(res, h1s, marker="o", linewidth=2.0, markersize=4.5, label="h1")
                ax.plot(res, h2s, marker="s", linewidth=2.0, markersize=4.5, label="h2")
                ax.set_ylabel("换热系数 h")
            else:
                ax.plot(res, nu1s, marker="o", linewidth=2.0, markersize=4.5, label="Nu1")
                ax.plot(res, nu2s, marker="s", linewidth=2.0, markersize=4.5, label="Nu2")
                ax.set_ylabel("努塞尔数 Nu")

            ax.set_xlabel("雷诺数 Re")
            ax.grid(True, alpha=0.28)
            ax.legend(frameon=False)
            self.fig.tight_layout()
            self.canvas.draw()

            self.status.setText(f"已绘制 {n_points} 个点，Re 范围 [{float(np.min(res)):.3g}, {float(np.max(res)):.3g}]")

    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    run_gui()
