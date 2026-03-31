# Launch of the application
import os
import sys
import tkinter as tk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.gui import SteganographyGUI

def main():
    root= tk.Tk()
    app = SteganographyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    #app = MainWindow()
    #app.run()
    main()