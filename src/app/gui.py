"""GUI entry point (see ui/gui_app.py for the actual application)."""

from __future__ import annotations


def main() -> int:
    import tkinter as tk

    from ui.gui_app import MotionToolApp

    root = tk.Tk()
    MotionToolApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
