import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    blue = "\x1b[34;20m"
    magenta = "\x1b[35;20m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = f"{self.green}[%(asctime)s]{self.reset} {self.blue}%(message)s{self.reset}"
        elif record.levelno == logging.WARNING:
            self._style._fmt = (
                f"{self.yellow}[%(asctime)s] WARNING: %(message)s{self.reset}"
            )
        elif record.levelno == logging.ERROR:
            self._style._fmt = f"{self.red}[%(asctime)s] ERROR: %(message)s{self.reset}"
        elif record.levelno == logging.DEBUG:
            self._style._fmt = (
                f"{self.grey}[%(asctime)s] DEBUG: %(message)s{self.reset}"
            )
        else:
            self._style._fmt = f"[%(asctime)s] %(message)s"

        self._style._fmt = f"{self.bold}[%(levelname)s]{self.reset} " + self._style._fmt
        return super().format(record)
