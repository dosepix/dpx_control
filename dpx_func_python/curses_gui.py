import curses
from curses import panel
import sys
import glob
import serial

class GUI:
    def __init__(self, items, stdscreen):
        self.window = stdscreen.subwin(0, 0)
        self.window.keypad(1)
        self.panel = panel.new_panel(self.window)
        self.panel.hide()
        panel.update_panels()

        self.position = 0
        self.items = items
        self.items.append(('exit', 'exit'))

    def navigate(self, n):
        self.position += n
        if self.position < 0:
            self.position = 0
        elif self.position >= len(self.items):
            self.position = len(self.items)-1

    def display(self):
        self.panel.top()
        self.panel.show()
        self.window.clear()

        while True:
            self.window.refresh()
            curses.doupdate()

            # Header
            self.window.addstr(2, 4, "DPX Control")
            self.window.addstr(4, 4, "Please select an option:", curses.A_BOLD)

            # Options
            for index, item in enumerate(self.items):
                if index == self.position:
                    mode = curses.A_REVERSE
                else:
                    mode = curses.A_NORMAL

                msg = '%d. %s' % (index + 1, item[0])
                self.window.addstr(6 + index, 4, msg, mode)

            # Get input
            key = self.window.getch()

            if key in [curses.KEY_ENTER, ord('\n')]:
                if self.position == len(self.items) - 1:
                    break
                else:
                    self.items[self.position][1]()

            elif key == curses.KEY_UP or key == ord('k'):
                self.navigate(-1)

            elif key == curses.KEY_DOWN or key == ord('j'):
                self.navigate(1)

            # Exit program
            elif key == ord('q'):
                break

        self.window.clear()
        self.panel.hide()
        panel.update_panels()
        curses.doupdate()

    # https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
    def serial_ports(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result
